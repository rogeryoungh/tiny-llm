# Tiny LLM

[中文](./README.zh.md)

A lightweight large language model inference engine implemented in C++, with support for Chinese (UTF-8) input.

> [!WARNING]
> 🚧 This project is still under active development.

## Features

- Pure C++ implementation, supports both CPU and CUDA inference
- Tokenizer uses a popcount trie for memory efficiency (inspired by [cloudflare/trie-hard](https://github.com/cloudflare/trie-hard))
- Compatible with Huggingface safetensors weight format
- Full UTF-8 support, suitable for Chinese scenarios
- Tested on models such as [Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B), [Qwen3-4B](https://huggingface.co/Qwen/Qwen2.5-3B/), and [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- Supports loading weights in bf16, fp16, or fp32 formats

## Usage

First, download the model and weights, then build the project. You can run it on CPU as follows:

```bash
git clone git@hf.co:Qwen/Qwen2.5-3B
xmake
# The executable will be generated at ./build/linux/x86_64/release/tiny-llm
xmake run tiny-llm -m ./Qwen2.5-3B --prompt "介绍一下杭州的美食。" --device=cpu
```

**For CUDA inference, weights must be converted to fp16 first.**

A Python script is provided to convert weights to bf16, fp16, or fp32:

```bash
uv init # Install dependencies
uv run scripts/convert.py --src ./Qwen2.5-3B/ --dst ./Qwen2.5-3B-fp16 --dtype fp16
xmake run tiny-llm -m ./Qwen2.5-3B-fp16 --prompt "介绍一下杭州的美食。" --device=cuda
```

## Benchmark

For performance benchmarking, I pad the prompt to a fixed length using spaces. The testing script can be found in scripts/benchmark.sh.

Test environment:

- CPU: AMD 9950x (16 cores, 32 threads)
- Memory: DDR5 48GB x2, 6400MHz CL32
- GPU: RTX 5070 Ti, GDDR7 16GB, 896 GB/s bandwidth, 256-bit bus

Benchmark parameters:

- kv-size=4096
- prompt=32
- max-output=512

### CUDA

The performance unit is token/s.

| Model Name               | Params | Precision | Sample | Prefill | Gen Avg | First32 | Last32 |
| ------------------------ | ----- | ---- | -------- | ------- | ------ | ------ | ------ |
| Mistral-7B-Instruct-v0.3 | 7.25B | fp16 | Argmax   | 54.65   | 51.62  | 53.25  | 51.36  |
| Qwen2.5-3B               | 3.09B | fp16 | Argmax   | 116.18  | 101.26 | 105.68 | 100.24 |
| Qwen2.5-7B-Instruct      | 7.62B | fp16 | Sampling | 57.27   | 52.04  | 53.17  | 52.39  |
| Qwen3-4B                 | 4.02B | fp16 | Sampling | 92.16   | 78.88  | 83.24  | 76.51  |
| Qwen3-0.6B               | 0.75B  | fp16 | Sampling | 376.89  | 283.99 | 314.57 | 268.62 |

### CPU

The performance unit is token/s.

| Model Name               | Params | Precision | Sample | Prefill | Gen Avg | First32 | Last32 |
| ------------------------ | ----- | ---- | -------- | ------- | ------ | ----- | ---- |
| Mistral-7B-Instruct-v0.3 | 7.25B | fp16 | Argmax   | 3.84    | 2.82   | 3.68  | 2.00 |
| Mistral-7B-Instruct-v0.3 | 7.25B | bf16 | Argmax   | 3.92    | 2.52   | 3.76  | 1.75 |
| Qwen2.5-3B               | 3.09B | fp16 | Argmax   | 7.47    | 6.18   | 6.90  | 5.80 |
| Qwen2.5-3B               | 3.09B | bf16 | Argmax   | 7.70    | 6.34   | 7.12  | 5.94 |
| Qwen2.5-7B-Instruct      | 7.62B | fp16 | Sampling | 4.06    | 3.45   | 3.86  | 3.20 |
| Qwen2.5-7B-Instruct      | 7.62B | bf16 | Sampling | 4.29    | 3.54   | 3.97  | 3.28 |
| Qwen3-4B                 | 4.02B | fp16 | Sampling | 6.22    | 3.92   | 5.55  | 2.61 |
| Qwen3-4B                 | 4.02B | bf16 | Sampling | 6.39    | 3.58   | 5.69  | 2.53 |
| Qwen3-0.6B               | 0.75B  | fp16 | Sampling | 21.35   | 10.38  | 18.10 | 6.10 |
| Qwen3-0.6B               | 0.75B  | bf16 | Sampling | 21.79   | 12.29  | 18.50 | 8.97 |
