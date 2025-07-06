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
| Mistral-7B-Instruct-v0.3 | 7.25B | fp16 | Argmax   | 54.56   | 51.62  | 53.16  | 51.28  |
| Qwen2.5-3B               | 3.09B | fp16 | Argmax   | 115.94  | 101.41 | 105.39 | 100.10 |
| Qwen2.5-7B-Instruct      | 7.62B | fp16 | Sampling | 57.52   | 51.98  | 53.07  | 52.31  |
| Qwen3-4B                 | 4.02B | fp16 | Sampling | 91.93   | 78.99  | 83.12  | 76.42  |
| Qwen3-0.6B               | 0.75B | fp16 | Sampling | 375.95  | 285.73 | 313.96 | 267.75 |

### CPU

The performance unit is token/s.

| Model Name               | Params | Precision | Sample | Prefill | Gen Avg | First32 | Last32 |
| ------------------------ | ----- | ---- | -------- | ------- | ------ | ----- | ---- |
| Mistral-7B-Instruct-v0.3 | 7.25B | fp16 | Argmax   | 4.38    | 4.17   | 4.24  | 4.25  |
| Mistral-7B-Instruct-v0.3 | 7.25B | bf16 | Argmax   | 4.38    | 4.18   | 4.25  | 4.25  |
| Qwen2.5-3B               | 3.09B | fp16 | Argmax   | 8.84    | 7.73   | 8.10  | 7.83  |
| Qwen2.5-3B               | 3.09B | bf16 | Argmax   | 8.80    | 7.76   | 8.08  | 7.95  |
| Qwen2.5-7B-Instruct      | 7.62B | fp16 | Sampling | 4.65    | 4.29   | 4.35  | 4.40  |
| Qwen2.5-7B-Instruct      | 7.62B | bf16 | Sampling | 4.70    | 4.28   | 4.34  | 4.39  |
| Qwen3-4B                 | 4.02B | fp16 | Sampling | 7.50    | 6.54   | 6.74  | 6.62  |
| Qwen3-4B                 | 4.02B | bf16 | Sampling | 7.48    | 6.52   | 6.72  | 6.63  |
| Qwen3-0.6B               | 0.75B | fp16 | Sampling | 24.69   | 19.30  | 21.54 | 18.69 |
| Qwen3-0.6B               | 0.75B | bf16 | Sampling | 24.93   | 19.71  | 21.69 | 19.32 |
