xmake
echo $LLM_PATH

xmake run tiny-llm -m $LLM_PATH/Mistral-7B-Instruct-v0.3-fp16 --benchmark-prompt-size=32 --max-tokens=512 --kv-size=4096 --device=cuda
xmake run tiny-llm -m $LLM_PATH/Mistral-7B-Instruct-v0.3-fp16 --benchmark-prompt-size=32 --max-tokens=512 --kv-size=4096 --device=cpu
xmake run tiny-llm -m $LLM_PATH/Mistral-7B-Instruct-v0.3 --benchmark-prompt-size=32 --max-tokens=512 --kv-size=4096 --device=cpu

xmake run tiny-llm -m $LLM_PATH/Qwen2.5-3B-fp16 --benchmark-prompt-size=32 --max-tokens=512 --kv-size=4096 --device=cuda
xmake run tiny-llm -m $LLM_PATH/Qwen2.5-3B-fp16 --benchmark-prompt-size=32 --max-tokens=512 --kv-size=4096 --device=cpu
xmake run tiny-llm -m $LLM_PATH/Qwen2.5-3B --benchmark-prompt-size=32 --max-tokens=512 --kv-size=4096 --device=cpu

xmake run tiny-llm -m $LLM_PATH/Qwen2.5-7B-Instruct-fp16 --benchmark-prompt-size=32 --max-tokens=512 --kv-size=4096 --device=cuda
xmake run tiny-llm -m $LLM_PATH/Qwen2.5-7B-Instruct-fp16 --benchmark-prompt-size=32 --max-tokens=512 --kv-size=4096 --device=cpu
xmake run tiny-llm -m $LLM_PATH/Qwen2.5-7B-Instruct --benchmark-prompt-size=32 --max-tokens=512 --kv-size=4096 --device=cpu

xmake run tiny-llm -m $LLM_PATH/Qwen3-4B-fp16 --benchmark-prompt-size=32 --max-tokens=512 --kv-size=4096 --device=cuda
xmake run tiny-llm -m $LLM_PATH/Qwen3-4B-fp16 --benchmark-prompt-size=32 --max-tokens=512 --kv-size=4096 --device=cpu
xmake run tiny-llm -m $LLM_PATH/Qwen3-4B --benchmark-prompt-size=32 --max-tokens=512 --kv-size=4096 --device=cpu

xmake run tiny-llm -m $LLM_PATH/Qwen3-0.6B-fp16 --benchmark-prompt-size=32 --max-tokens=512 --kv-size=4096 --device=cuda
xmake run tiny-llm -m $LLM_PATH/Qwen3-0.6B-fp16 --benchmark-prompt-size=32 --max-tokens=512 --kv-size=4096 --device=cpu
xmake run tiny-llm -m $LLM_PATH/Qwen3-0.6B --benchmark-prompt-size=32 --max-tokens=512 --kv-size=4096 --device=cpu
