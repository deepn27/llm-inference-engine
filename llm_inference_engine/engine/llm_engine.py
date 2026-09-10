import atexit
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import fields
from time import perf_counter
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from llm_inference_engine.config import Config
from llm_inference_engine.sampling_params import SamplingParams
from llm_inference_engine.engine.sequence import Sequence
from llm_inference_engine.engine.scheduler import Scheduler
from llm_inference_engine.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.enable_metrics = config.enable_metrics
        self.enable_nvtx = config.enable_nvtx
        Sequence.block_size = config.kvcache_block_size
        self.model_runner = ModelRunner(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        self.max_model_len = config.max_model_len
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        self.metrics = {"requests": [], "steps": [], "summary": {}}
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.exit()
        del self.model_runner

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        arrival_time = perf_counter()
        tokenization_start = perf_counter()
        if isinstance(prompt, str):
            nvtx_range = torch.cuda.nvtx.range("tokenization") if self.enable_nvtx else nullcontext()
            with nvtx_range:
                prompt = self.tokenizer.encode(prompt)
        tokenization_time = perf_counter() - tokenization_start
        request_len = len(prompt) + sampling_params.max_tokens
        if request_len > self.max_model_len:
            raise ValueError(f"Request length {request_len} exceeds max_model_len {self.max_model_len}")
        seq = Sequence(prompt, sampling_params, arrival_time, tokenization_time)
        self.scheduler.add(seq)

    def step(self):
        step_start = perf_counter()
        nvtx_range = torch.cuda.nvtx.range("scheduler") if self.enable_nvtx else nullcontext()
        with nvtx_range:
            seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        mode = "prefill" if is_prefill else "decode"
        step_label = f"engine_step.{mode}:batch={len(seqs)},tokens={abs(num_tokens)}"
        nvtx_range = torch.cuda.nvtx.range(step_label) if self.enable_nvtx else nullcontext()
        with nvtx_range:
            token_ids = self.model_runner.run(seqs, is_prefill)
            self.scheduler.postprocess(seqs, token_ids)
        if self.enable_metrics:
            self.metrics["steps"].append({
                "index": len(self.metrics["steps"]),
                "duration_ms": (perf_counter() - step_start) * 1000,
                "scheduler": dict(self.scheduler.last_metrics),
                "model_runner": dict(self.model_runner.last_metrics),
            })
            self.metrics["requests"].extend(self._request_metrics(seq) for seq in seqs if seq.is_finished)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def _request_metrics(self, seq: Sequence):
        output_tokens = seq.num_completion_tokens
        return {
            "seq_id": seq.seq_id,
            "prompt_tokens": seq.num_prompt_tokens,
            "output_tokens": output_tokens,
            "tokenization_ms": seq.tokenization_time * 1000,
            "waiting_ms": (seq.running_since - seq.waiting_since) * 1000,
            "running_ms": (seq.finished_time - seq.running_since) * 1000,
            "ttft_ms": (seq.first_token_time - seq.arrival_time) * 1000,
            "tpot_ms": ((seq.finished_time - seq.first_token_time) * 1000 / (output_tokens - 1)
                        if output_tokens > 1 else 0.0),
            "end_to_end_ms": (seq.finished_time - seq.arrival_time) * 1000,
            "detokenization_ms": 0.0,
        }

    def get_metrics(self):
        if not self.enable_metrics:
            return self.metrics
        requests = self.metrics["requests"]
        steps = self.metrics["steps"]
        request_fields = (
            "tokenization_ms",
            "waiting_ms",
            "running_ms",
            "ttft_ms",
            "tpot_ms",
            "end_to_end_ms",
            "detokenization_ms",
        )
        request_averages = {
            name: sum(request[name] for request in requests) / len(requests)
            for name in request_fields
        } if requests else {}
        scheduler = {
            "total_ms": sum(step["scheduler"]["duration_ms"] for step in steps),
            "average_waiting_depth": (sum(step["scheduler"]["waiting_depth"] for step in steps) / len(steps)
                                      if steps else 0.0),
            "average_running_depth": (sum(step["scheduler"]["running_depth"] for step in steps) / len(steps)
                                      if steps else 0.0),
            "average_batch_size": (sum(step["scheduler"]["batch_size"] for step in steps) / len(steps)
                                   if steps else 0.0),
        }
        cpu_ms = defaultdict(float)
        gpu_ms = defaultdict(float)
        for step in steps:
            for name, duration in step["model_runner"]["cpu_ms"].items():
                cpu_ms[name] += duration
            for name, duration in step["model_runner"]["gpu_ms"].items():
                gpu_ms[name] += duration
        self.metrics["summary"].update({
            "request_count": len(requests),
            "step_count": len(steps),
            "prefill_steps": sum(step["model_runner"]["mode"] == "prefill" for step in steps),
            "decode_steps": sum(step["model_runner"]["mode"] == "decode" for step in steps),
            "peak_waiting_depth": max((step["scheduler"]["waiting_depth"] for step in steps), default=0),
            "peak_batch_size": max((step["model_runner"]["batch_size"] for step in steps), default=0),
            "prompt_tokens": sum(item["prompt_tokens"] for item in requests),
            "output_tokens": sum(item["output_tokens"] for item in requests),
            "request_average_ms": request_averages,
            "scheduler": scheduler,
            "model_runner_total_cpu_ms": dict(cpu_ms),
            "model_runner_total_gpu_ms": dict(gpu_ms),
        })
        return self.metrics

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        self.metrics = {"requests": [], "steps": [], "summary": {}}
        generation_start = perf_counter()
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        sorted_outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        decoded_outputs = []
        request_metrics = {item["seq_id"]: item for item in self.metrics["requests"]}
        for seq_id, token_ids in zip(sorted(outputs.keys()), sorted_outputs):
            detokenization_start = perf_counter()
            nvtx_range = torch.cuda.nvtx.range("detokenization") if self.enable_nvtx else nullcontext()
            with nvtx_range:
                text = self.tokenizer.decode(token_ids)
            if self.enable_metrics:
                request_metrics[seq_id]["detokenization_ms"] = (perf_counter() - detokenization_start) * 1000
            decoded_outputs.append({"text": text, "token_ids": token_ids})
        if self.enable_metrics:
            self.metrics["summary"]["total_duration_ms"] = (perf_counter() - generation_start) * 1000
            self.get_metrics()
        return decoded_outputs
