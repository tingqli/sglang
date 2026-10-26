

# 最新客户要求启动方式

```bash
sglang serve \
  --trust-remote-code \
  --model-path $MODEL_PATH \
  --served-model-name XiaomiMiMo/MiMo-V2.6-Flash-RL \
  --tp 8 \
  --attn-dp-size 2 \
  --moe-runner-backend auto \
  --mm-enable-dp-encoder \
  --mem-fraction-static 0.80 \
  --chunked-prefill-size 16384 \
  --reasoning-parser mimo \
  --tool-call-parser mimo \
  --host 0.0.0.0 \
  --port 30000 2>&1 | tee sglang_$(date +%Y%m%d_%H%M%S).log



curl --fail-with-body -sS http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "XiaomiMiMo/MiMo-V2.6-Flash-RL",
    "messages": [
      {"role": "user", "content": "What is the capital of France? Answer briefly."}
    ],
    "max_completion_tokens": 64,
    "temperature": 0
  }' | python3 -m json.tool



sgl-eval run gsm8k \
  --base-url http://127.0.0.1:30000/v1 \
  --num-examples 1319 \
  --temperature 1.0 \
  --top-p 0.95 \
  --max-tokens 4096 \
  --thinking \
  --num-threads 64


./aiter/configs/model_configs/mimo_v2.6_flash_a16w4_untuned_fmoe.csv

# 根据https://huggingface.co/tiyuvta/MiMo-V2.6-Flash-RL-NVFP4#1来看
# 社区精度显式 MiMo-V2.6-Flash-RL 在 GSM8K test 下 1271/1319 (96.36%)，a16w4 可以复现精度

sgl-eval run gsm8k \
  --base-url http://127.0.0.1:30000/v1 \
  --num-examples 1319 \
  --temperature 1.0 \
  --top-p 0.95 \
  --max-tokens 4096 \
  --thinking \
  --num-threads 64

gsm8k: 100%|███████████████████████████████████████| 1319/1319 [12:55<00:00,  1.70it/s, acc=96.36%, truncated=10/1319 (0.76%)]
== gsm8k ==
1319 examples (single-shot)  |  775.2s  |  524 tok/s  |  407K tokens

* score                =  96.36%
  stop_rate            =  99.24%
  truncated_rate       =  0.76%
  error_rate           =  0.00%
  avg_input_tokens     =  91.3 tokens
  avg_output_tokens    =  308.2 tokens
  avg_thinking_tokens  =  0.0 tokens

```

