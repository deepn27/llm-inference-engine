from collections import deque
from time import perf_counter

from llm_inference_engine.config import Config
from llm_inference_engine.engine.sequence import Sequence, SequenceStatus
from llm_inference_engine.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.eos = config.eos
        self.max_batch_tokens = config.max_batch_tokens
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.last_metrics = {}

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        seq.waiting_since = perf_counter()
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        schedule_start = perf_counter()
        waiting_depth = len(self.waiting)

        scheduled_seqs = []
        scheduled_tokens = 0
        while self.waiting:
            seq = self.waiting[0]
            remaining_budget = self.max_batch_tokens - scheduled_tokens
            if remaining_budget == 0:
                break
            if not seq.block_table:
                if not self.block_manager.can_allocate(seq):
                    if not self.running and not scheduled_seqs:
                        raise RuntimeError("Insufficient KV cache for request")
                    break
                self.block_manager.allocate(seq)
            remaining_tokens = seq.num_tokens - seq.num_cached_tokens
            if scheduled_seqs and remaining_tokens > remaining_budget:
                break
            seq.num_scheduled_tokens = min(remaining_tokens, remaining_budget)
            scheduled_seqs.append(seq)
            scheduled_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                self.waiting.popleft()
                seq.status = SequenceStatus.RUNNING
                seq.running_since = perf_counter()
                self.running.append(seq)

        if scheduled_seqs:
            self.last_metrics = {
                "duration_ms": (perf_counter() - schedule_start) * 1000,
                "waiting_depth": waiting_depth,
                "running_depth": len(self.running) - len(scheduled_seqs),
                "batch_size": len(scheduled_seqs),
                "scheduled_tokens": scheduled_tokens,
                "is_prefill": True,
            }
            return scheduled_seqs, True

        for seq in self.running:
            if not self.block_manager.can_append(seq):
                raise RuntimeError("Insufficient KV cache to continue request")
            seq.num_scheduled_tokens = 1
            seq.is_prefill = False
            self.block_manager.may_append(seq)
        self.last_metrics = {
            "duration_ms": (perf_counter() - schedule_start) * 1000,
            "waiting_depth": waiting_depth,
            "running_depth": len(self.running),
            "batch_size": len(self.running),
            "scheduled_tokens": len(self.running),
            "is_prefill": False,
        }
        return self.running.copy(), False

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        for seq, token_id in zip(seqs, token_ids):
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if seq.is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if not seq.first_token_time:
                seq.first_token_time = perf_counter()
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                seq.finished_time = perf_counter()
                self.block_manager.deallocate(seq)
        self.running = [seq for seq in self.running if not seq.is_finished]
