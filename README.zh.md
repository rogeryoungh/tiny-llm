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

## 性能测试

为便于性能测试，我使用空格将 prompt 填充至固定长度。具体的测试脚本见 `scripts/benchmark.sh`。

测试环境如下：

- CPU：AMD 9950x（16 核 32 线程）
- 内存：DDR5 48GB x2，6400MHz CL32
- 显卡：RTX 5070 Ti，GDDR7 16GB，带宽 896 GB/s，256 bit 总线

测试参数：

- kv-size=4096
- prompt=32
- max-output=512

### CUDA

性能单位是 token/s。

| 模型名称                 | 参数量  | 精度   | 采样     | Prefill | Gen 平均 | 首32    | 末32    |
| ------------------------ | ----- | ---- | -------- | ------- | ------ | ------ | ------ |
| Mistral-7B-Instruct-v0.3 | 7.25B | fp16 | Argmax   | 54.56   | 51.62  | 53.16  | 51.28  |
| Qwen2.5-3B               | 3.09B | fp16 | Argmax   | 115.94  | 101.41 | 105.39 | 100.10 |
| Qwen2.5-7B-Instruct      | 7.62B | fp16 | Sampling | 57.52   | 51.98  | 53.07  | 52.31  |
| Qwen3-4B                 | 4.02B | fp16 | Sampling | 91.93   | 78.99  | 83.12  | 76.42  |
| Qwen3-0.6B               | 0.75B | fp16 | Sampling | 375.95  | 285.73 | 313.96 | 267.75 |

### CPU

性能单位是 token/s。


| 模型名称                 | 参数量  | 精度   | 采样     | Prefill | Gen 平均 | 首32   | 末32  |
| ------------------------ | ----- | ---- | -------- | ------- | ------ | ----- | ----- |
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
