import torch
import torch.nn as nn
from typing import Optional
from einops import rearrange
import torch.nn.functional as F

from kv_cache import KV_cache
from model_config import ModelConfig
from tensor_tools import init_custom_weight
from rotary_pos_enc import apply_rotary_pos_emb


class Attention_SubBlock(nn.Module):

    def __init__(self, cfg:ModelConfig):
        super().__init__()

        self.embed_dim    = cfg.embed_dim
        self.num_q_heads  = cfg.attention.num_q_heads
        self.num_kv_heads = cfg.attention.num_kv_heads
        self.num_layers   = cfg.num_layers
        
        if self.embed_dim % self.num_q_heads != 0:
            raise ValueError("embed_dim must be divisible by num_q_heads")
        if self.num_q_heads % self.num_kv_heads != 0:
            raise ValueError("num_q_heads must be divisible by num_kv_heads")

        self.head_dim = self.embed_dim // self.num_q_heads
        self.q_dim    = self.num_q_heads * self.head_dim
        self.kv_dim   = self.num_kv_heads * self.head_dim

        # ONE matrix for Q, K and V: columns are [ q | k | v ]
        self.W_qkv = nn.Parameter(torch.empty(self.embed_dim, self.q_dim + 2 * self.kv_dim))
        self.W_o   = nn.Parameter(torch.empty(self.q_dim, self.embed_dim))

        self.init_params()



    def init_params(self):

        q_end = self.q_dim
        k_end = q_end + self.kv_dim

        W_q = self.W_qkv[:, :q_end]
        W_k = self.W_qkv[:, q_end:k_end]
        W_v = self.W_qkv[:, k_end:]

        init_custom_weight(W_q, is_residual_output=False)
        init_custom_weight(W_k, is_residual_output=False)
        init_custom_weight(W_v, is_residual_output=False)

        # Residual output projection (scaled by depth)
        init_custom_weight(self.W_o, 
                           is_residual_output=True, 
                           num_layers=self.num_layers)


    def forward(
        self,
        x:   torch.Tensor,
        cos: torch.Tensor,          # [B, 1, seq, head_dim] – already computed
        sin: torch.Tensor,
        cache: Optional[KV_cache] = None,
        layer_idx: int = 0,
        
        lane_ids:  Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,   # [B, 1, T, kv_len] bool, from cache.prepare()
   
    ) -> torch.Tensor:

        batch_size, seq_len, embed_dim = x.shape

        qkv = x @ self.W_qkv  # [B, T, q_dim + 2*kv_dim]
        q, k, v = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)


        q = rearrange(q, 
                      "batch_size seq_len (num_q_heads head_dim)" 
                      " -> " 
                      "batch_size num_q_heads seq_len head_dim", 
                      num_q_heads=self.num_q_heads)
        
        k = rearrange(k, 
                      "batch_size seq_len (num_kv_heads head_dim)" 
                      " -> " 
                      "batch_size num_kv_heads seq_len head_dim",
                      num_kv_heads=self.num_kv_heads)
        
        v = rearrange(v, 
                      "batch_size seq_len (num_kv_heads head_dim)" 
                      " -> " 
                      "batch_size num_kv_heads seq_len head_dim",
                      num_kv_heads=self.num_kv_heads)


        q, k = apply_rotary_pos_emb(q, k, cos, sin)


        if cache is not None:
            k, v = cache.update(layer_idx, lane_ids, k, v)



        # The mask rule is that a query at absolute position p may attend to a key at 
        # absolute position a iff p - W < a <= p. The <= p part is causality, 
        # and the > p - W part is the sliding window. 
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=(attn_mask is None), # fast path only when no mask is given
            enable_gqa=(self.num_q_heads != self.num_kv_heads),
        )


        # Merge attention heads
        out = rearrange(
            out,
            "batch_size num_q_heads seq_len head_dim"
            " -> " 
            "batch_size seq_len (num_q_heads head_dim)",
        )


        return out @ self.W_o
