# Tiny LLM

一个使用 C++ 实现的大模型推理引擎，支持中文（UTF-8）输入。

> [!WARNING]
> 🚧 项目仍在持续开发中。

## 特性

- 主要使用 C++ 实现
- 支持 Huggingface 的 safetensors 格式
- 支持 UTF-8 编码，适合中文场景
- 已在 [Qwen/Qwen2.5-3B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/)、[Qwen3-4B](https://huggingface.co/Qwen/Qwen2.5-3B/) 和 [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B/) 等模型上进行测试
- 支持模型权重以 bf16 或 fp32 格式加载

## 当前限制

- 仅支持 CPU 推理

## 使用

首先下载模型和权重，然后编译运行。

```bash
git clone git@hf.co:Qwen/Qwen2.5-3B
xmake
xmake run tiny-llm ./Qwen2.5-3B
```

如需将模型权重转换为 fp32 或 bf16，可使用提供的 Python 脚本：

```bash
uv init # 安装依赖
uv run scripts/convert.py --src ./Qwen2.5-3B/ --dst ./Qwen2.5-3B-fp32 --dtype fp32
xmake run tiny-llm ./Qwen2.5-3B-fp32
```
