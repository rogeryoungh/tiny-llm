# Tiny LLM

一个使用 C++ 实现的大模型推理引擎，目前支持基本的中文（UTF-8）输入。

> [!WARNING]
> 尚在测试中。

## 特性

- 主要使用 C++ 实现
- 支持 UTF-8 编码，适用于中文文本（功能仍在完善）

## 当前限制

- 仅测试 Mistral-7B-Instruct-v0.2
- 仅实现了 fp32 精度
- 仅支持 CPU

## 使用

当前仅在 [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/) 上测试。

首先下载模型和权重，然后安装 python 环境并把模型转换成 fp32（暂不支持其他精度）。

```bash
git clone git@hf.co:mistralai/Mistral-7B-Instruct-v0.2
uv init
uv run scripts/convert.py --src ./Mistral-7B-Instruct-v0.2/ --dst ./Mistral-7B-Instruct-v0.2-fp32 --dtype fp32
```

然后编译运行

```bash
xmake
xmake run tiny-llm ./Mistral-7B-Instruct-v0.2-fp32
```
