

# MiMo-V2.6 MXFP4 -> FP8 smoke test (MI308X)

保留 TP8、attention DP2、AITER 和 CUDA graph。`SGLANG_DSV4_FP4_DEQUANT`
在加载阶段将 expert 转成 block-FP8；MoE 后端保持 `auto`。

```bash
# 使用 FP4 -> FP8 路径
export SGLANG_USE_AITER=1
export SGLANG_DSV4_FP4_DEQUANT=1

# 调试时遇到 NaN 直接报错，不把 HTTP 200 当作数值正确
export SGLANG_ENABLE_NAN_LOGITS_CHECK=1

export MODEL_PATH="/raid/models/MiMo-V2.6-Flash-MOPD/"
export MODEL_PATH="/raid/models/MiMo-V2.6-Flash-RL"
# 启动
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

# 服务就绪后，在另一终端运行 smoke test
curl --fail-with-body -sS http://localhost:30001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "XiaomiMiMo/MiMo-V2.6-Flash-RL",
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
```
## 遗留问题 -- 精度测试

```bash
pip install sgl-eval
sgl-eval run gsm8k \
  --base-url http://127.0.0.1:30000/v1 \
  --num-examples 200

# sglang.test.few_shot_gsm8k 测试不太准确
python3 -m sglang.test.few_shot_gsm8k \
    --host 127.0.0.1 \
    --port 30000 \
    --num-questions 2000 \
    --parallel 128 \
    --num-shots 5
Accuracy: 0.886
Invalid: 0.000
Latency: 250.924 s
Output throughput: 731.995 token/s

# --num-shots 5 代表 Few-shot（少样本）提示中的示例数量。在你的命令中，它表示在向模型提问每一道 GSM8K 测试题之前，会先在提示词（prompt）里拼接 5 个 带有标准答案的示例问答对，以此引导模型模仿解题格式和推理方式。测试脚本会从 GSM8K 数据集中取出前 5 道题（包含题目和详细解答），将它们作为“范例”拼接到一起，形成一段固定的前缀文本。型收到实际的测试题目时，它看到的完整输入是：[5个示例问答] + [当前需要回答的题目]。这样做的目的是让模型学会按照“先写推理步骤，最后给出 #### 答案”的格式来生成回答，从而提高答案提取的准确率。这 5 个示例通常来自数据集的开头，为了避免数据泄露，实际的评估题目通常会从这 5 个示例之后开始选取（或者在取测试题时逻辑上跳过它们），以确保测试的公平性。SGLang 的相关测试脚本和文档中，--num-shots 的默认值通常就是 5。

# 根据https://huggingface.co/tiyuvta/MiMo-V2.6-Flash-RL-NVFP4#1来看
# 社区精度显式 MiMo-V2.6-Flash-RL 在 GSM8K test 下 1271/1319 (96.36%)，我们的结果显然偏低
# GSM8K used temperature 1.0, top-p 0.95, thinking enabled, a 4,096-token cap and concurrency 16.
# 参考 https://lmsysorg.mintlify.app/docs/developer_guide/evaluating_new_models#llms
# 改为使用下面的命令
sgl-eval run gsm8k \
  --base-url http://127.0.0.1:30000/v1 \
  --num-examples 1319 \
  --temperature 1.0 \
  --top-p 0.95 \
  --max-tokens 4096 \
  --thinking

gsm8k: 100%|███████████████████████████████████████████████████████| 1319/1319 [11:34<00:00,  1.90it/s, acc=96.06%, truncated=8/1319 (0.61%)]
== gsm8k ==
1319 examples (single-shot)  |  694.3s  |  591 tok/s  |  411K tokens
* score                =  96.06%
  stop_rate            =  99.39%
  truncated_rate       =  0.61%
  error_rate           =  0.00%
  avg_input_tokens     =  91.3 tokens
  avg_output_tokens    =  311.2 tokens
  avg_thinking_tokens  =  0.0 tokens

# 不使用额外设定，仅仅使用 sgl-eval, （--max-tokens 不加的话会导致thinking过程死循环，最后的几道题让模型无法结束，但似乎正确率会高一些）
sgl-eval run gsm8k \
  --base-url http://127.0.0.1:30000/v1 \
  --num-examples 1319 \
  --max-tokens 4096

gsm8k: 100%|████████████████████████████████████████████████████████| 1319/1319 [11:36<00:00,  1.89it/s, acc=96.06%, truncated=18/1319 (1.36%)]
* score                =  96.06%
  stop_rate            =  98.64%
  truncated_rate       =  1.36%  [warn: hitting max_tokens]
  error_rate           =  0.00%
  avg_input_tokens     =  91.3 tokens
  avg_output_tokens    =  321.0 tokens
  avg_thinking_tokens  =  0.0 tokens


hf download XiaomiMiMo/MiMo-V2.6-Flash-RL --local-dir MiMo-V2.6-Flash-RL --token=hf_...

```


https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD#sglang
```bash
# 使用兼容客户要求的方式启动：
sglang serve \
  --trust-remote-code \
  --model-path $MODEL_PATH \
  --served-model-name XiaomiMiMo/MiMo-V2.6-Flash-RL \
  --tp-size 4 \
  --dp-size 2 \
  --ep-size 1 \
  --moe-runner-backend auto \
  --mem-fraction-static 0.80 \
  --chunked-prefill-size 16384 \
  --reasoning-parser mimo \
  --tool-call-parser mimo \
  --host 0.0.0.0 \
  --port 30000 2>&1 | tee sglang_$(date +%Y%m%d_%H%M%S).log

sgl-eval run gsm8k \
  --base-url http://127.0.0.1:30000/v1 \
  --num-examples 1319 \
  --temperature 1.0 \
  --top-p 0.95 \
  --max-tokens 4096 \
  --thinking \
  --num-threads 64


* score                =  96.06%
  stop_rate            =  99.32%
  truncated_rate       =  0.68%
  error_rate           =  0.00%
  avg_input_tokens     =  91.3 tokens
  avg_output_tokens    =  312.8 tokens
  avg_thinking_tokens  =  0.0 tokens
```

## 遗留问题 -- 使用mxfp4
```bash
root@banff-csq-dh2-be08-14:/sgl-workspace/aiter# git remote add tq ...tingqli/aiter.git
root@banff-csq-dh2-be08-14:/sgl-workspace/aiter# git fetch tq tq/gfx942-a16w4-a8w4


# 切换aiter的基础版本为xiake的
# 安装后回归测试基本情况

token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1
1,4096,512,256,8,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fnuz,torch.float8_e4m3fnuz,QuantType.per_128x128,1,0
2,4096,512,256,8,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fnuz,torch.float8_e4m3fnuz,QuantType.per_128x128,1,0
4,4096,512,256,8,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fnuz,torch.float8_e4m3fnuz,QuantType.per_128x128,1,0
8,4096,512,256,8,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fnuz,torch.float8_e4m3fnuz,QuantType.per_128x128,1,0
16,4096,512,256,8,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fnuz,torch.float8_e4m3fnuz,QuantType.per_128x128,1,0
32,4096,512,256,8,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fnuz,torch.float8_e4m3fnuz,QuantType.per_128x128,1,0
64,4096,512,256,8,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fnuz,torch.float8_e4m3fnuz,QuantType.per_128x128,1,0
128,4096,512,256,8,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fnuz,torch.float8_e4m3fnuz,QuantType.per_128x128,1,0
256,4096,512,256,8,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fnuz,torch.float8_e4m3fnuz,QuantType.per_128x128,1,0
512,4096,512,256,8,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fnuz,torch.float8_e4m3fnuz,QuantType.per_128x128,1,0
1024,4096,512,256,8,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fnuz,torch.float8_e4m3fnuz,QuantType.per_128x128,1,0
2048,4096,512,256,8,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fnuz,torch.float8_e4m3fnuz,QuantType.per_128x128,1,0
4096,4096,512,256,8,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fnuz,torch.float8_e4m3fnuz,QuantType.per_128x128,1,0
8192,4096,512,256,8,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fnuz,torch.float8_e4m3fnuz,QuantType.per_128x128,1,0
16384,4096,512,256,8,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fnuz,torch.float8_e4m3fnuz,QuantType.per_128x128,1,0
32768,4096,512,256,8,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fnuz,torch.float8_e4m3fnuz,QuantType.per_128x128,1,0
```