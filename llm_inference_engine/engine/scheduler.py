from collections import deque
from time import perf_counter

from llm_inference_engine.config import Config
from llm_inference_engine.engine.sequence import Sequence, SequenceStatus
from llm_inference_engine.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: Sequence | None = None
        self.last_metrics = {}

    def is_finished(self):
        return not self.waiting and self.running is None

    def add(self, seq: Sequence):
        seq.waiting_since = perf_counter()
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        schedule_start = perf_counter()
        waiting_depth = len(self.waiting)
        if self.running is None:
            seq = self.waiting.popleft()
            if not self.block_manager.can_allocate(seq):
                raise RuntimeError("Insufficient KV cache for request")
            self.block_manager.allocate(seq)
            seq.num_scheduled_tokens = seq.num_tokens
            seq.status = SequenceStatus.RUNNING
            seq.running_since = perf_counter()
            self.running = seq
            self.last_metrics = {
                "duration_ms": (perf_counter() - schedule_start) * 1000,
                "waiting_depth": waiting_depth,
                "running_depth": 0,
                "batch_size": 1,
                "scheduled_tokens": seq.num_scheduled_tokens,
                "is_prefill": True,
            }
            return [seq], True

        seq = self.running
        if not self.block_manager.can_append(seq):
            raise RuntimeError("Insufficient KV cache to continue request")
        seq.num_scheduled_tokens = 1
        seq.is_prefill = False
        self.block_manager.may_append(seq)
        self.last_metrics = {
            "duration_ms": (perf_counter() - schedule_start) * 1000,
            "waiting_depth": waiting_depth,
            "running_depth": 1,
            "batch_size": 1,
            "scheduled_tokens": 1,
            "is_prefill": False,
        }
        return [seq], False

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        for seq, token_id in zip(seqs, token_ids):
            seq.num_scheduled_tokens = 0
            seq.append_token(token_id)
            if not seq.first_token_time:
                seq.first_token_time = perf_counter()
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                seq.finished_time = perf_counter()
                self.block_manager.deallocate(seq)
                self.running = None
