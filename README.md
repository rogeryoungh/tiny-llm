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

Download model and weights, build, and run. `bf16` may need 30 seconds to convert.

```bash
git clone git@hf.co:mistralai/Mistral-7B-Instruct-v0.2
xmake
xmake run tiny-llm ./Mistral-7B-Instruct-v0.2
```

If you need fp32 weights file, run the python script.

```bash
uv init # 安装依赖
uv run scripts/convert.py --src ./Mistral-7B-Instruct-v0.2/ --dst ./Mistral-7B-Instruct-v0.2-fp32 --dtype fp32
xmake run tiny-llm ./Mistral-7B-Instruct-v0.2-fp32
```
