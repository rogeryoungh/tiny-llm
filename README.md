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
