import torch
import torch.nn as nn
from typing import Tuple
from model_config import ModelConfig


class RotaryEmbedding(nn.Module):

    def __init__(self, cfg:ModelConfig):
        super().__init__()

        base     = cfg.attention.rope_theta
        head_dim = cfg.embed_dim // cfg.attention.num_q_heads


        # shape: [head_dim // 2]
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        ).to(dtype=torch.float32)
        
        self.register_buffer("inv_freq", inv_freq, persistent=False)


    def forward(
            self,
            position_ids: torch.Tensor,   # [B, seq] or [seq]
            dtype= torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
            
            """position_ids: [B, T] (or [T]) -> cos, sin of shape [B, 1, T, head_dim]"""

            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)          # [1, seq]


            # position_ids: [B, seq]  (integer)
            # inv_freq:     [head_dim//2]
            # freqs:        [B, seq, head_dim//2]
            freqs = torch.einsum(
                "bi,j->bij",
                position_ids.to(dtype=self.inv_freq.dtype),
                self.inv_freq,
            )

            emb = torch.cat((freqs, freqs), dim=-1) # [B, seq, head_dim]

            # broadcast over the head dimension → [B, 1, seq, head_dim]
            return emb.cos().to(dtype).unsqueeze(1), emb.sin().to(dtype).unsqueeze(1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q:   torch.Tensor,
    k:   torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    cos = cos.to(device=q.device, dtype=q.dtype)
    sin = sin.to(device=q.device, dtype=q.dtype)

    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k