from typing import Iterable, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from transformers import AutoTokenizer

model = "/sgl-workspace/guppylm-9M"
tokenizer = AutoTokenizer.from_pretrained(
    model,
    trust_remote_code=True,
)

class SimpleTorchBlockCausalAttention(nn.Module):
    """Torch SDPA attention with KV-cache read/write via ForwardBatch pools.

    This follows the same cache access pattern as `TorchNativeAttnBackend`:
    - write current-step `k, v` into `token_to_kv_pool` with `out_cache_loc`
    - gather full per-request KV history from radix cache (`req_to_token_pool`)
    - run SDPA per request
    """

    def __init__(self, num_heads: int, head_dim: int, scaling: float, layer_id: int):
        super().__init__()
        self.tp_q_head_num = num_heads
        self.tp_k_head_num = num_heads
        self.tp_v_head_num = num_heads
        self.qk_head_dim = head_dim
        self.v_head_dim = head_dim
        self.scaling = scaling
        self.layer_id = layer_id
        self.is_cross_attention = False
        self.attn_type = AttentionType.DECODER

    def _run_sdpa_forward_extend(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        extract_kv,
    ) -> None:
        # [num_tokens, num_heads, head_dim] -> [num_heads, num_tokens, head_dim]
        query = query.movedim(0, query.dim() - 2)
        start_q = 0
        for seq_idx in range(seq_lens.shape[0]):
            extend_seq_len_q = int(extend_seq_lens[seq_idx].item())
            prefill_seq_len_q = int(extend_prefix_lens[seq_idx].item())
            seq_len_kv = int(seq_lens[seq_idx].item())
            end_q = start_q + extend_seq_len_q

            per_req_query = query[:, start_q:end_q, :]
            per_req_query_redundant = torch.empty(
                (per_req_query.shape[0], seq_len_kv, per_req_query.shape[2]),
                dtype=per_req_query.dtype,
                device=per_req_query.device,
            )
            per_req_query_redundant[:, prefill_seq_len_q:, :] = per_req_query

            per_req_key, per_req_value = extract_kv(seq_idx)
            per_req_out_redundant = (
                F.scaled_dot_product_attention(
                    per_req_query_redundant.unsqueeze(0), #N,...,Hq,L,E
                    per_req_key.unsqueeze(0),             #N,...,H,S,E
                    per_req_value.unsqueeze(0),           #N,...,H,S,Ev
                    enable_gqa=False,
                    scale=self.scaling,
                    is_causal=True,
                )
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )
            output[start_q:end_q, :, :] = per_req_out_redundant[prefill_seq_len_q:, :, :]
            start_q = end_q

    def _run_sdpa_forward_decode(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        seq_lens: torch.Tensor,
        extract_kv,
    ) -> None:
        # [num_tokens, num_heads, head_dim] -> [num_heads, num_tokens, head_dim]
        query = query.movedim(0, query.dim() - 2)
        
        for seq_idx in range(seq_lens.shape[0]):
            per_req_query = query[:, seq_idx:seq_idx+1, :]

            per_req_key, per_req_value = extract_kv(seq_idx)
            per_req_out = (
                F.scaled_dot_product_attention(
                    per_req_query.unsqueeze(0), #N,...,Hq,L,E
                    per_req_key.unsqueeze(0),   #N,...,H,S,E
                    per_req_value.unsqueeze(0), #N,...,H,S,Ev
                    enable_gqa=False,
                    scale=self.scaling,
                    is_causal=False,
                )
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )
            # [num_seq, num_heads, head_dim]
            output[seq_idx:seq_idx+1, :, :] = per_req_out


    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs  # RoPE/sink kwargs are ignored in this SDPA reference path.
        if forward_batch.forward_mode.is_idle():
            return q.new_empty(q.shape[0], self.tp_q_head_num * self.v_head_dim)
        if self.qk_head_dim != self.v_head_dim:
            o = q.new_empty((q.shape[0], self.tp_q_head_num * self.v_head_dim))
        else:
            o = torch.empty_like(q)

        if self.is_cross_attention:
            cache_loc = forward_batch.encoder_out_cache_loc
        else:
            cache_loc = forward_batch.out_cache_loc
        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(self, cache_loc, k, v)

        q_ = q.view(-1, self.tp_q_head_num, self.qk_head_dim)
        o_ = o.view(-1, self.tp_q_head_num, self.v_head_dim)
        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(self.layer_id)
        v_cache = forward_batch.token_to_kv_pool.get_value_buffer(self.layer_id)

        def extract_kv(seq_idx):
            seq_len_kv = int(forward_batch.seq_lens[seq_idx].item())
            req_pool_idx = forward_batch.req_pool_indices[seq_idx]
            per_req_tokens = forward_batch.req_to_token_pool.req_to_token[req_pool_idx, :seq_len_kv]
            # KV cache [max_tokens_in_cache, num_heads, head_dim]  with num_heads = num_total_heads // tp_size
            # gather gives [seq_len, num_heads, head_dim] (token-major);
            # movedim -> [num_heads, seq_len, head_dim] to match SDPA's (N, L, E).
            per_req_key = k_cache[per_req_tokens].movedim(0, q_.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, q_.dim() - 2)

            if not (q_.dtype == per_req_key.dtype == per_req_value.dtype):
                per_req_key = per_req_key.to(q_.dtype)
                per_req_value = per_req_value.to(q_.dtype)
            return per_req_key, per_req_value

        if forward_batch.forward_mode.is_decode():
            if self.layer_id == 0 and get_tensor_model_parallel_rank() == 0:
                print("================== decode ")
                for index,(ids,pos) in enumerate(zip(forward_batch.input_ids.tolist(), forward_batch.seq_lens.tolist())):
                    text = tokenizer.decode(ids, skip_special_tokens=False)
                    print(f"\t[{index}]  {pos} : {text.encode('utf-8')}")
            self._run_sdpa_forward_decode(
                q_,
                o_,
                forward_batch.seq_lens,
                extract_kv
            )
        else:
            if self.layer_id == 0 and get_tensor_model_parallel_rank() == 0:
                print("================== extend ")
                start_q = 0
                for seq_idx in range(forward_batch.seq_lens.shape[0]):
                    extend_seq_len_q = int(forward_batch.extend_seq_lens[seq_idx].item())
                    prefill_seq_len_q = int(forward_batch.extend_prefix_lens[seq_idx].item())
                    seq_len_kv = int(forward_batch.seq_lens[seq_idx].item())
                    end_q = start_q + extend_seq_len_q
                    text = tokenizer.decode(forward_batch.input_ids[start_q:end_q], skip_special_tokens=False)
                    start_q = end_q
                    print(f"\t [{seq_idx}] {prefill_seq_len_q} + {extend_seq_len_q} == {seq_len_kv} : ", text.encode('utf-8'))

            self._run_sdpa_forward_extend(
                q_,
                o_,
                forward_batch.seq_lens,
                forward_batch.extend_prefix_lens,
                forward_batch.extend_seq_lens,
                extract_kv
            )
        return o


class Attention(nn.Module):
    def __init__(self, config, layer_id: int):
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.n_heads
        assert self.total_num_heads % tp_size == 0
        self.n_heads = self.total_num_heads // tp_size        
        self.head_dim = config.d_model // self.total_num_heads
        self.layer_id = layer_id
        self.qkv = QKVParallelLinear( #nn.Linear(config.d_model, 3 * config.d_model)
                config.d_model,
                self.head_dim,
                self.total_num_heads,
                self.total_num_heads,
                bias=True,
                quant_config=None,
            )
        self.out = RowParallelLinear( # nn.Linear(config.d_model, config.d_model)
                self.total_num_heads * self.head_dim,
                config.d_model,
                bias=True,
                quant_config=None,
            )
        self.dropout = nn.Dropout(config.dropout)
        self.scaling = self.head_dim**-0.5
        if getattr(config, "simple_torch_attention", True):
            self.attn = SimpleTorchBlockCausalAttention(
                self.n_heads, self.head_dim, self.scaling, layer_id
            )
        else:
            self.attn = RadixAttention(
                self.n_heads,
                self.head_dim,
                self.scaling,
                num_kv_heads=self.n_heads,
                layer_id=layer_id,
                quant_config=None,
            )

    def forward(self, x: torch.Tensor, forward_batch: ForwardBatch):
        BT, C = x.shape
        qkv, _ = self.qkv(x)
        qkv = qkv.reshape(BT, 3, self.n_heads*self.head_dim).permute(1, 0, 2)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_out = self.attn(q, k, v, forward_batch)
        return self.out(attn_out)[0].contiguous().view(BT, C)


class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.up = ColumnParallelLinear(
            config.d_model,
            config.ffn_hidden,
            bias=True,
            quant_config=None,
        )#nn.Linear(config.d_model, config.ffn_hidden)
        self.down = RowParallelLinear(
            config.ffn_hidden,
            config.d_model,
            bias=True,
            quant_config=None,
            reduce_results=True
        ) #nn.Linear(config.ffn_hidden, config.d_model)

    def forward(self, x: torch.Tensor):
        up_out, _ = self.up(x)
        return self.down(F.relu(up_out))[0]


class Block(nn.Module):
    def __init__(self, config, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.norm1 = nn.LayerNorm(config.d_model)
        self.self_attn = Attention(config, layer_id)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ffn = FFN(config)

    def forward(self, x: torch.Tensor, forward_batch: ForwardBatch):
        x = x + self.self_attn(self.norm1(x), forward_batch)
        x = x + self.ffn(self.norm2(x))
        return x

class GuppyLMModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.layers = nn.ModuleList([Block(config, i) for i in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        # input_ids & positions are both 1D tensors with ragged concat layout
        x = self.tok_emb(input_ids) + self.pos_emb(positions)

        for block in self.layers:
            x = block(x, forward_batch)

        hidden_states = self.norm(x)

        return hidden_states

class GuppyLMForCausalLM(nn.Module):
    def __init__(
        self,
        config,
        quant_config = None,
        prefix: str = "",
    ):
        super().__init__()
        assert quant_config is None
        self.config = config
        self.quant_config = quant_config
        self.prefix = prefix
        self.model = GuppyLMModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.model.tok_emb.weight
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids,
        positions,
        forward_batch,
    ):
        hidden_states = self.model(input_ids, positions, forward_batch)

        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            my_name = name.replace("model.blocks.", "model.layers.")
            my_name = my_name.replace(".attn.", ".self_attn.")
            param = params_dict[my_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)

EntryClass = [GuppyLMForCausalLM]
