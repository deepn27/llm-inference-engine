from collections import deque

from llm_inference_engine.config import Config
from llm_inference_engine.engine.sequence import Sequence, SequenceStatus
from llm_inference_engine.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: Sequence | None = None

    def is_finished(self):
        return not self.waiting and self.running is None

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        if self.running is None:
            seq = self.waiting.popleft()
            if not self.block_manager.can_allocate(seq):
                raise RuntimeError("Insufficient KV cache for request")
            self.block_manager.allocate(seq)
            seq.num_scheduled_tokens = seq.num_tokens
            seq.status = SequenceStatus.RUNNING
            self.running = seq
            return [seq], True

        seq = self.running
        if not self.block_manager.can_append(seq):
            raise RuntimeError("Insufficient KV cache to continue request")
        seq.num_scheduled_tokens = 1
        seq.is_prefill = False
        self.block_manager.may_append(seq)
        return [seq], False

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        for seq, token_id in zip(seqs, token_ids):
            seq.num_scheduled_tokens = 0
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running = None
