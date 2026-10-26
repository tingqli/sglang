#!/usr/bin/env bash

PORT="${PORT:-30000}"
MODEL="${MODEL:-MiMo-V2.6-Flash-RL}"

curl --fail-with-body -sS http://localhost:${PORT}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "XiaomiMiMo/MiMo-V2.6-Flash-RL",
    "messages": [
      {"role": "user", "content": "What is the capital of France? Answer briefly."}
    ],
    "max_completion_tokens": 64,
    "temperature": 0
  }' | python3 -m json.tool
