# Tiny LLM

[中文](./README.zh.md)

A lightweight large language model inference engine implemented in C++.

> [!WARNING]
> 🚧 This project is still under active development.

## Feature

- Written primarily in C++
- Supports Huggingface safetensors format
- Supports UTF-8 input (Such as Chinese)
- Tested with models such as [Qwen/Qwen2.5-3B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/), [Qwen3-4B](https://huggingface.co/Qwen/Qwen2.5-3B/), [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- Model weights can be loaded in bf16 or fp32 formats

## Limitations

- Inference is currently supported on CPU only

## Usage

Download model and weights, compile and run the program.

```bash
git clone git@hf.co:Qwen/Qwen2.5-3B
xmake
xmake run tiny-llm ./Qwen2.5-3B
```

If you need to convert model weights to fp32 or bf16, a Python script is also provided:

```bash
uv init # install dependencies
uv run scripts/convert.py --src ./Qwen2.5-3B/ --dst ./Qwen2.5-3B-fp32 --dtype fp32
xmake run tiny-llm ./Qwen2.5-3B-fp32
```
