from collections import defaultdict

from llm_inference_engine.engine.sequence import Sequence


class EngineMetrics:

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.reset()

    def reset(self):
        self.data = {"requests": [], "steps": [], "summary": {}}

    def record_step(self, duration_ms: float, scheduler: dict, model_runner: dict):
        if not self.enabled:
            return
        self.data["steps"].append({
            "index": len(self.data["steps"]),
            "duration_ms": duration_ms,
            "scheduler": dict(scheduler),
            "model_runner": dict(model_runner),
        })

    def record_request(self, seq: Sequence):
        if not self.enabled:
            return
        output_tokens = seq.num_completion_tokens
        self.data["requests"].append({
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
        })

    def record_detokenization(self, seq_id: int, duration_ms: float):
        if not self.enabled:
            return
        for request in self.data["requests"]:
            if request["seq_id"] == seq_id:
                request["detokenization_ms"] = duration_ms
                return

    def finalize(self, total_duration_ms: float):
        if not self.enabled:
            return
        requests = self.data["requests"]
        steps = self.data["steps"]
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
            "average_waiting_depth": self._step_average(steps, "waiting_depth"),
            "average_running_depth": self._step_average(steps, "running_depth"),
            "average_batch_size": self._step_average(steps, "batch_size"),
        }
        cpu_ms = defaultdict(float)
        gpu_ms = defaultdict(float)
        for step in steps:
            for name, duration in step["model_runner"]["cpu_ms"].items():
                cpu_ms[name] += duration
            for name, duration in step["model_runner"]["gpu_ms"].items():
                gpu_ms[name] += duration
        self.data["summary"] = {
            "total_duration_ms": total_duration_ms,
            "request_count": len(requests),
            "step_count": len(steps),
            "prefill_steps": sum(step["model_runner"]["mode"] == "prefill" for step in steps),
            "decode_steps": sum(step["model_runner"]["mode"] == "decode" for step in steps),
            "peak_waiting_depth": max((step["scheduler"]["waiting_depth"] for step in steps), default=0),
            "peak_batch_size": max((step["model_runner"]["batch_size"] for step in steps), default=0),
            "prompt_tokens": sum(request["prompt_tokens"] for request in requests),
            "output_tokens": sum(request["output_tokens"] for request in requests),
            "request_average_ms": request_averages,
            "scheduler": scheduler,
            "model_runner_total_cpu_ms": dict(cpu_ms),
            "model_runner_total_gpu_ms": dict(gpu_ms),
        }

    @staticmethod
    def _step_average(steps: list[dict], name: str):
        return sum(step["scheduler"][name] for step in steps) / len(steps) if steps else 0.0

    def get(self):
        return self.data