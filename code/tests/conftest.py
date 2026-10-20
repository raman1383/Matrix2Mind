import sys
import importlib
from pathlib import Path

# flat-module repo: make `import model_config`, `import kv_cache`, ... work from tests/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch

from model_config import ModelConfig, AttentionConfig, FFNConfig


# (module, class) pairs, tried in order. Add yours here if you rename the file again.
_MODEL_CANDIDATES = (
    ("transformer_model", "Transformer_Model"),
)


def model_class():
    """
    The model class. A PLAIN import, so a missing module FAILS the test instead of
    silently skipping it (importorskip would hide a broken test suite as "skipped").
    Errors raised INSIDE a found module (a typo in attention.py, say) propagate untouched.
    """
    for module_name, class_name in _MODEL_CANDIDATES:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            if e.name != module_name:
                raise                      # a different import inside the module failed
            continue
        return getattr(module, class_name)
    raise ImportError(
        "No model module found. Tried: "
        + ", ".join(f"{m}.{c}" for m, c in _MODEL_CANDIDATES)
    )


def make_cfg(heads=4, kv=2, layers=2, experts=1, top_k=1, cf=None,
             window=16, gate=True) -> ModelConfig:
    return ModelConfig(
        embed_dim=32, vocab_size=64, max_context_window=window, num_layers=layers,
        tied_embeddings=False,
        attention=AttentionConfig(num_q_heads=heads, num_kv_heads=kv, rope_theta=10000.0),
        ffn=FFNConfig(
            use_gate=gate, mlp_to_embed_expand_factor=2,
            MoE_num_experts=experts, MoE_num_experts_per_tkn=top_k,
            MoE_capacity_factor=cf,
            MoE_router_aux_loss_coef=0.01, MoE_router_z_loss_coeff=0.0,
        ),
    )


def randomize(module: torch.nn.Module, seed: int = 0, std: float = 0.3):
    """
    Your init (std=0.02) makes logits ~0, so almost any bug still passes allclose.
    Large random weights make outputs depend strongly on every code path.
    """
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            if p.ndim >= 2:
                p.copy_(torch.randn(p.shape, generator=g, dtype=p.dtype) * std)
    return module


def build_model(cfg: ModelConfig, seed: int = 0):
    """float64 + random weights + eval mode."""
    return randomize(model_class()(cfg).double(), seed).eval()


def tokens(n: int, vocab: int = 64, seed: int = 1, batch: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (batch, n), generator=g)