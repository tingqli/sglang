

# MiMo-V2.6 MXFP4 -> FP8 smoke test (MI308X)

保留 TP8、attention DP2、AITER 和 CUDA graph。`SGLANG_DSV4_FP4_DEQUANT`
在加载阶段将 expert 转成 block-FP8；MoE 后端保持 `auto`。

```bash
# 使用 FP4 -> FP8 路径
export SGLANG_USE_AITER=1
export SGLANG_DSV4_FP4_DEQUANT=1

# 调试时遇到 NaN 直接报错，不把 HTTP 200 当作数值正确
export SGLANG_ENABLE_NAN_LOGITS_CHECK=1

# 启动
sglang serve \
  --trust-remote-code \
  --model-path /tingqli/MiMo-V2.6-Flash-MOPD \
  --served-model-name XiaomiMiMo/MiMo-V2.6-Flash-MOPD \
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

# 服务就绪后，在另一终端运行 smoke test
curl --fail-with-body -sS http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "XiaomiMiMo/MiMo-V2.6-Flash-MOPD",
    "messages": [
      {"role": "user", "content": "What is the capital of France? Answer briefly."}
    ],
    "max_completion_tokens": 64,
    "temperature": 0
  }' | python3 -m json.tool

```

## 本次修复

- 错误输出：`The!!!!!!!!...`，后续 token 的 logprob 非有限。
- 首次 NaN 出现在第 0 层 attention，Q/K/V 输入均有限，尚未经过 MoE。
- MiMo 的 K head dimension 为 192、V 为 128。融合 KV 写入内核共用 K 的
  宽度寻址和掩码，导致 V 跨 head 读取，并覆盖相邻 cache 槽位。
- [KV 写入内核](python/sglang/kernels/ops/kvcache/cache_ops.py) 现为 K/V 使用
  独立宽度和掩码；[回归测试](test/registered/kernels/ops/kvcache/test_reshape_and_cache_flash.py)
  同时检查目标值和未写入槽位。
- [FP8 权重处理](python/sglang/srt/layers/quantization/fp8.py) 保留实际 shuffle
  后的 `is_shuffled` 标记，补齐 separated gate/up 元数据，删除临时打印和死分支。

同一 TP8/attention DP2 配置、CUDA graph 开启、NaN 检查开启后，原始请求返回：
**The capital of France is Paris.**，`finish_reason=stop`。

额外验证通过：3 次重复请求、4 路并发请求、558-token 长 prompt；所有响应
正常结束，生成 logprob 均为有限值。新增 KV 写入回归测试修复前失败、修复后
通过；原有 7 项 KV 测试及 6 项相关 FP8 布局测试通过。完整 FP8 单测文件中的
另外两项使用 Python 3.11 的 `enterContext`，本机 Python 3.10 无法运行，未改动。

本机诊断材料保存在 `/tmp/sglang_mimo_fp4_debug/`，修复后服务日志为
`/tmp/sglang_mimo_fp4_fixed.log`。这些是本次调试产物，不随仓库提交。


## 人工复测

log里面出现了下面的字样，说明 128 blockscale的aiter fused_moe kernel被调用
```bash
[aiter] [fused_moe] using 2stage default for ('gfx942', 80, 8, 4096, 256, 256, 8, <ActivationType.Silu: 0>, 'torch.bfloat16', 'torch.float8_e4m3fnuz', 'torch.float8_e4m3fnuz', 'QuantType.per_1x128', True, False) 

curl -L --fail -o ShareGPT_V3_unfiltered_cleaned_split.json \
  https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

python3 -m sglang.bench_serving \
  --backend sglang \
  --model /tingqli/MiMo-V2.6-Flash-MOPD \
  --host 127.0.0.1 --port 30000 \
  --dataset-name sharegpt \
  --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
  --flush-cache --seed 12345 \
  --num-prompts 256 --warmup-requests 4 --max-concurrency 4 \
  --temperature 0 \
  --output-file baseline_results.jsonl --output-details \
  --tag baseline_no_spec


python3 -m sglang.test.few_shot_gsm8k \
  --host 127.0.0.1 \
  --port 30000 \
  --num-questions 1319 \
  --num-shots 5 \
  --parallel 4 \
  --max-new-tokens 512 \
  --temperature 0

python3 -m sglang.test.few_shot_gsm8k \
    --host 127.0.0.1 \
    --port 30000 \
    --num-questions 2000 \
    --parallel 128 \
    --num-shots 5

pip install sgl-eval
sgl-eval run gsm8k \
  --base-url http://127.0.0.1:30000/v1 \
  --num-examples 200


```

## 遗留问题
https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD#sglang
```bash
sglang serve \
  --trust-remote-code \
  --model-path XiaomiMiMo/MiMo-V2.6-Flash-MOPD \
  --tp 8 \
  --dp 2 \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --mm-enable-dp-encoder \
  --mem-fraction-static 0.65 \
  --chunked-prefill-size 16384 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --enable-multi-layer-eagle \
  --reasoning-parser mimo \
  --tool-call-parser mimo \
  --host 0.0.0.0 \
  --port 30000

```