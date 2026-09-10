# LLM Inference Engine

A lightweight LLM inference engine built from scratch.

## Key Features

* 🚀 **Offline inference** - Requests execute sequentially on a single GPU
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimized execution** - FlashAttention and paged KV caching

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/llm-inference-engine.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from llm_inference_engine import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH")
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, LLM Inference Engine."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

## License

See [LICENSE](LICENSE).