
TUNE_ONLY=flydsl python csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py \
  -i ./aiter/configs/model_configs/mimo_v2.6_flash_a16w4_untuned_fmoe.csv \
  -o ./aiter/configs/model_configs/mimo_v2.6_flash_a16w4_tuned_fmoe.csv \
  --mp 8 --warmup 10 --iters 51 --timeout 480 --all

