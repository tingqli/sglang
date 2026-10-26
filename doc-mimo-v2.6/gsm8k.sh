#!/usr/bin/env bash
PORT="${PORT:-30000}"

sgl-eval run gsm8k \
  --base-url http://127.0.0.1:${PORT}/v1 \
  --num-examples 1319 \
  --temperature 1.0 \
  --top-p 0.95 \
  --max-tokens 4096 \
  --thinking \
  --num-threads 64
