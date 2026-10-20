import torch
from torch import nn

from kv_cache import KV_cache
from typing import Optional, Tuple
from ffn import build_ffn_sub_block
from model_config import ModelConfig
from attention import Attention_SubBlock
from rotary_pos_enc import RotaryEmbedding
from tensor_tools import init_custom_weight


class Transformer_Block(nn.Module):
    """
    Pre-norm block:
        x = x + Attention(RMSNorm(x))
        x = x + FFN(RMSNorm(x))
    Returns (x, aux_loss).
    """
 
    def __init__(self, cfg: ModelConfig):
        super().__init__()
 
        self.norm1 = nn.RMSNorm(cfg.embed_dim)
        self.norm2 = nn.RMSNorm(cfg.embed_dim)
 
        self.attention_sub_block = Attention_SubBlock(cfg)
        self.ffn_sub_block       = build_ffn_sub_block(cfg)
 
    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: Optional[KV_cache] = None,
        layer_idx: int = 0,
        lane_ids=None,
        attn_mask=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
 
        x = x + self.attention_sub_block(
            self.norm1(x),
            cos, sin,
            cache=cache,
            layer_idx=layer_idx,
            lane_ids=lane_ids,
            attn_mask=attn_mask,
        )
 
        ffn_out, aux_loss = self.ffn_sub_block(self.norm2(x))
        x = x + ffn_out
 
        return x, aux_loss




class Transformer_Model(nn.Module):
 
    def __init__(self, cfg: ModelConfig):
        super().__init__()
 
        self.vocab_size = cfg.vocab_size
        self.embed_dim  = cfg.embed_dim
        self.tied_embeddings = cfg.tied_embeddings
 
        self.embed = nn.Parameter(torch.empty(self.vocab_size, self.embed_dim))
 
        if not self.tied_embeddings:
            self.unembed = nn.Parameter(torch.empty(self.embed_dim, self.vocab_size))
 
        self.rotary = RotaryEmbedding(cfg)
 
        self.layers = nn.ModuleList(
            [Transformer_Block(cfg) for _ in range(cfg.num_layers)]
        )
 
        self.final_rms_norm = nn.RMSNorm(cfg.embed_dim)
 
        self.init_params()
 
    def init_params(self):
        init_custom_weight(self.embed, is_residual_output=False)
 
        if not self.tied_embeddings:
            init_custom_weight(self.unembed, is_residual_output=False)
 
    def forward(
        self,
        input_ids: torch.Tensor,
        cache: Optional[KV_cache] = None,
        lane_ids: Optional[torch.Tensor] = None,   # which cache lane each batch row uses
        attn_mask=None,                            # only used without cache (tests)
        return_aux_loss: bool = False,
    ):
        """
        Default: returns logits [B, T, V]  (drop-in compatible with the training / inference
        engines you already have).
        return_aux_loss=True: returns (logits, aux_loss). aux_loss is the MoE router loss
        averaged over layers, ALREADY scaled by its coefficients, so the training step is
        simply   loss = cross_entropy + aux_loss   (it is 0.0 for dense models).
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device                  # from the data, never a global
 
        x = self.embed[input_ids]                  # [B, T] -> [B, T, D]
 
        if cache is None:
            position_ids = torch.arange(seq_len, device=device).expand(batch_size, -1)
            # attn_mask stays whatever the caller passed (None -> is_causal fast path)
        else:
            if lane_ids is None:
                lane_ids = torch.arange(batch_size)
            position_ids, attn_mask = cache.prepare(lane_ids, seq_len)
 
        cos, sin = self.rotary(position_ids, dtype=x.dtype)
 
        aux_total = torch.zeros((), dtype=torch.float32, device=device)
 
        for i, layer in enumerate(self.layers):
            x, aux = layer(
                x, cos, sin,
                cache=cache,
                layer_idx=i,
                lane_ids=lane_ids,
                attn_mask=attn_mask,
            )
            aux_total = aux_total + aux
 
        if cache is not None:
            cache.commit(lane_ids, seq_len)        # once, after ALL layers
 
        x = self.final_rms_norm(x)
 
        logits = x @ self.embed.T if self.tied_embeddings else x @ self.unembed
 
        if return_aux_loss:
            return logits, aux_total / len(self.layers)
        return logits
 
