from typing import Iterable, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.radix_attention import RadixAttention

class Attention(nn.Module):
    def __init__(self, config, layer_id: int):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.layer_id = layer_id
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.scaling = self.head_dim**-0.5
        self.attn = RadixAttention(
            self.n_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.n_heads,
            layer_id=layer_id,
            quant_config=None,
        )

    def forward(self, x, forward_batch):

        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads*self.head_dim).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_out = self.attn(q, k, v, forward_batch)
        return self.out(attn_out).contiguous().view(B, T, C)


class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.up = nn.Linear(config.d_model, config.ffn_hidden)
        self.down = nn.Linear(config.ffn_hidden, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.down(F.relu(self.up(x))))


class Block(nn.Module):
    def __init__(self, config, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn = Attention(config, layer_id)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ffn = FFN(config)

    def forward(self, x, forward_batch):
        x = x + self.attn(self.norm1(x), forward_batch)
        x = x + self.ffn(self.norm2(x))
        return x

class GuppyLMModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config, i) for i in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        input_ids,
        positions,
        forward_batch
    ):
        do_unsqueeze = False
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
            assert positions.ndim == 1
            positions = positions.unsqueeze(0)
            do_unsqueeze = True

        #print(input_ids.shape, positions.shape)
        #print(input_ids, positions)
        pos = positions
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))

        for block in self.blocks:
            x = block(x, forward_batch)

        hidden_states = self.norm(x)

        return hidden_states.squeeze(0) if do_unsqueeze else hidden_states

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

        # logits = self.lm_head(hidden_states)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)

EntryClass = [GuppyLMForCausalLM]
