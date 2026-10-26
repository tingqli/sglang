#!/usr/bin/env bash

MODEL_PATH="${MODEL_PATH:-/raid/models/MiMo-V2.6-Flash-RL}"
unset SGLANG_DSV4_FP4_DEQUANT
sglang serve \
  --trust-remote-code \
  --model-path $MODEL_PATH \
  --served-model-name XiaomiMiMo/MiMo-V2.6-Flash-RL \
  --tp 8 \
  --attn-dp-size 2 \
  --moe-runner-backend aiter \
  --mm-enable-dp-encoder \
  --mem-fraction-static 0.80 \
  --chunked-prefill-size 16384 \
  --reasoning-parser mimo \
  --tool-call-parser mimo \
  --host 0.0.0.0 \
  --port 30000 2>&1 | tee sglang_$(date +%Y%m%d_%H%M%S).log

