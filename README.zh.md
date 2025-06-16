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
| Mistral-7B-Instruct-v0.3 | 7.25B | fp16 | Argmax   | 54.65   | 51.62  | 53.25  | 51.36  |
| Qwen2.5-3B               | 3.09B | fp16 | Argmax   | 116.18  | 101.26 | 105.68 | 100.24 |
| Qwen2.5-7B-Instruct      | 7.62B | fp16 | Sampling | 57.27   | 52.04  | 53.17  | 52.39  |
| Qwen3-4B                 | 4.02B | fp16 | Sampling | 92.16   | 78.88  | 83.24  | 76.51  |
| Qwen3-0.6B               | 0.75B  | fp16 | Sampling | 376.89  | 283.99 | 314.57 | 268.62 |

### CPU

性能单位是 token/s。


| 模型名称                 | 参数量  | 精度   | 采样     | Prefill | Gen 平均 | 首32   | 末32  |
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
