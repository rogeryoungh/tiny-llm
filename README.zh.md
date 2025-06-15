# Tiny LLM

一个使用 C++ 实现的大模型推理引擎，支持中文（UTF-8）输入。

> [!WARNING]
> 🚧 项目仍在持续开发中。

## 特性

- 纯 C++ 实现，支持 CPU、CUDA 推理
- Tokenizer 采用 popcount trie 优化，内存占用低，借鉴 [cloudflare/trie-hard](https://github.com/cloudflare/trie-hard)
- 兼容 Huggingface safetensors 权重格式
- 支持 UTF-8 编码，适合中文场景
- 已在 [Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B)、[Qwen3-4B](https://huggingface.co/Qwen/Qwen2.5-3B/) 和 [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) 等模型上进行测试
- 支持以 bf16、fp16 或 fp32 格式加载权重

## 使用

首先下载模型和权重，然后编译，即可在 CPU 上运行。

```bash
git clone git@hf.co:Qwen/Qwen2.5-3B
xmake
# 生成的可执行文件位于 ./build/linux/x86_64/release/tiny-llm
xmake run tiny-llm -m ./Qwen2.5-3B --prompt "介绍一下杭州的美食。" --device=cpu
```

**在 CUDA 上运行需要先转化为 fp16。**

我提供了一个 Python 脚本用于把权重转化为 bf16、fp16 或 fp32。

```bash
uv init # 安装依赖
uv run scripts/convert.py --src ./Qwen2.5-3B/ --dst ./Qwen2.5-3B-fp16 --dtype fp16
xmake run tiny-llm -m ./Qwen2.5-3B-fp16 --prompt "介绍一下杭州的美食。" --device=cuda
```
