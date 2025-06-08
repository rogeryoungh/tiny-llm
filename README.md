# Tiny LLM

[中文](./README.zh.md)

A large language model inference in C++.

> [!WARNING]
> Still working in process.

## Feature

- Written primarily in C++
- Supports UTF-8 input (Chinese support is experimental)

## Limitations

- Only supports Mistral-7B-Instruct-v0.2
- FP32 precision only
- CPU only

## Usage

Only test for [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/).

Download model and weights, set up Python environment, and convert the weights to fp32

```bash
git clone git@hf.co:mistralai/Mistral-7B-Instruct-v0.2
uv init
uv run scripts/convert.py --src ./Mistral-7B-Instruct-v0.2/ --dst ./Mistral-7B-Instruct-v0.2-fp32 --dtype fp32
```

Build and run:

```bash
xmake
xmake run tiny-llm ./Mistral-7B-Instruct-v0.2-fp32
```
