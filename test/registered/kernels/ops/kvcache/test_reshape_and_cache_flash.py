"""Paged KV writes must preserve independent K/V widths and untouched slots."""

import unittest

import torch

from sglang.kernels.ops.kvcache.cache_ops import launch_reshape_and_cache_flash
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")
# backend-specific: AITER's SWA writer consumes asymmetric BF16 K/V on ROCm.
register_amd_ci(est_time=15, suite="stage-b-test-1-gpu-small-amd")


@unittest.skipUnless(torch.cuda.is_available(), "requires a GPU")
class TestReshapeAndCacheFlash(CustomTestCase):
    def _check_write(self, *, key_dim, value_dim, heads, page_size, hnd, scaled, swa):
        tokens, num_blocks = 4, 32
        row_size = heads * (2 * key_dim + value_dim)
        # Guard rows make a regressed source overread deterministic and in-bounds.
        qkv = (
            torch.arange((tokens + 2) * row_size, device="cuda")
            .remainder(61)
            .reshape(tokens + 2, row_size)
            .to(torch.bfloat16)
        )
        _, key, value = qkv[:tokens].split(
            [heads * key_dim, heads * key_dim, heads * value_dim], dim=-1
        )
        key = key.view(tokens, heads, key_dim)
        value = value.view(tokens, heads, value_dim)
        cache_dtype = torch.bfloat16
        if scaled:
            cache_dtype = (
                torch.float8_e4m3fnuz if torch.version.hip else torch.float8_e4m3fn
            )

        def cache(dim):
            shape = (num_blocks, page_size, heads, dim)
            if hnd:
                shape = (num_blocks, heads, page_size, dim)
            storage = torch.full(shape, -7.0, device="cuda", dtype=cache_dtype)
            return storage.transpose(1, 2) if hnd else storage

        key_cache, value_cache = cache(key_dim), cache(value_dim)
        expected_key, expected_value = key_cache.clone(), value_cache.clone()
        slots = torch.tensor([1, 5, 9, 13], device="cuda", dtype=torch.int64)
        swa_mapping = None
        target_slots = slots
        if swa:
            swa_mapping = torch.arange(
                num_blocks * page_size, device="cuda", dtype=torch.int64
            )
            target_slots = torch.tensor(
                [18, 3, 17, 6], device="cuda", dtype=torch.int64
            )
            swa_mapping[slots] = target_slots
        key_scale = torch.tensor([0.5], device="cuda") if scaled else None
        value_scale = torch.tensor([0.25], device="cuda") if scaled else None

        key_ref, value_ref = key.float(), value.float()
        if scaled:
            key_ref = key_ref / key_scale
            value_ref = value_ref / value_scale
        block_ids, offsets = target_slots // page_size, target_slots % page_size
        expected_key[block_ids, offsets] = key_ref.to(cache_dtype)
        expected_value[block_ids, offsets] = value_ref.to(cache_dtype)

        launch_reshape_and_cache_flash(
            key=key,
            value=value,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=slots,
            swa_slot_mapping=swa_mapping,
            k_scale=key_scale,
            v_scale=value_scale,
        )

        torch.testing.assert_close(
            key_cache.float(), expected_key.float(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            value_cache.float(), expected_value.float(), rtol=0, atol=0
        )

    def test_asymmetric_bf16_preserves_values_and_neighbor_slots(self):
        """A wider K must not overwrite neighboring V slots or misindex V heads."""
        for key_dim, value_dim in ((192, 128), (128, 192)):
            for heads in (1, 2, 5):
                for page_size, hnd in ((1, False), (16, True)):
                    with self.subTest(
                        key_dim=key_dim, value_dim=value_dim, heads=heads, hnd=hnd
                    ):
                        self._check_write(
                            key_dim=key_dim,
                            value_dim=value_dim,
                            heads=heads,
                            page_size=page_size,
                            hnd=hnd,
                            scaled=False,
                            swa=True,
                        )

    def test_scaled_writes_preserve_independent_value_width(self):
        for key_dim, value_dim in ((192, 128), (128, 192), (128, 128)):
            with self.subTest(key_dim=key_dim, value_dim=value_dim):
                self._check_write(
                    key_dim=key_dim,
                    value_dim=value_dim,
                    heads=5,
                    page_size=16,
                    hnd=False,
                    scaled=True,
                    swa=False,
                )


if __name__ == "__main__":
    unittest.main()
