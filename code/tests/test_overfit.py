# tests/test_overfit.py
#
# TEST 4: the model must be able to memorise ONE small batch.
# If it can't, something is broken in the forward/backward path (dead gradient,
# wrong mask, detached residual, bad init, router blocking learning, ...),
# no matter how good the parity tests look.

import math

import pytest
import torch
import torch.nn.functional as F

from conftest import make_cfg, tokens


def overfit(cfg, steps=500, lr=1e-2, seed=0):
    mod = pytest.importorskip("transformer_model")
    torch.manual_seed(seed)
    model = mod.Transformer_Model(cfg).train()      # DEFAULT init, float32, train mode

    ids = tokens(17, vocab=cfg.vocab_size, batch=4)         # 4 x 16 targets, fits in the window
    x, y = ids[:, :-1], ids[:, 1:]

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    first = None
    for _ in range(steps):
        logits, aux = model(x, return_aux_loss=True)
        ce = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        if first is None:
            first = ce.item()
        opt.zero_grad(set_to_none=True)
        (ce + aux).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    with torch.no_grad():
        model.eval()                                        # drop-free MoE, no aux
        last = F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
    return first, last


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("dense_mha",  dict(heads=1, kv=1, experts=1)),
        ("dense_gqa",  dict(heads=4, kv=2, experts=1)),
        ("moe_top2",   dict(heads=4, kv=2, experts=4, top_k=2, cf=None)),  # cf=None: nothing dropped
    ],
)
def test_overfit_one_batch(name, kwargs):
    cfg = make_cfg(**kwargs)
    first, last = overfit(cfg)

    # 1) init sanity: an untrained model should be ~uniform, loss ~ ln(V)
    assert first == pytest.approx(math.log(cfg.vocab_size), abs=0.15), \
        f"{name}: initial loss {first:.3f} is far from ln(V)={math.log(cfg.vocab_size):.3f}"

    # 2) it must be able to memorise 64 target tokens
    assert last < 0.05, f"{name}: could not overfit one batch (loss {first:.3f} -> {last:.3f})"