# tests/test_moe.py
#
# TEST 2: the vectorised MoE (one-hot / cumsum / index_add / bmm) must equal an
# obviously-correct per-token Python loop, and must actually train.

import pytest
import torch
import torch.nn.functional as F

from conftest import make_cfg, build_model, randomize, tokens

ATOL = 1e-8


def naive_moe(blk, x):
    """Slow reference: for every token, run its top-k experts one at a time."""
    flat = x.reshape(-1, x.shape[-1])
    H = blk.hidden_dim

    # The block routes in float32 on purpose, so the reference must too, or
    # near-ties could pick different experts and we'd test noise instead of logic.
    probs = (flat.float() @ blk.router.float()).softmax(-1)
    w, idx = probs.topk(blk.top_k, dim=-1)
    if blk.top_k > 1:
        w = w / w.sum(-1, keepdim=True)

    out = torch.zeros_like(flat)
    for t in range(flat.shape[0]):
        for s in range(blk.top_k):
            e = idx[t, s]
            h = flat[t] @ blk.W_in[e]
            h = F.silu(h[:H]) * h[H:] if blk.use_gate else F.silu(h)
            out[t] += w[t, s] * (h @ blk.W_down[e])
    return out.view_as(x)


def make_block(E, k, gate):
    ffn = pytest.importorskip("ffn")                  # MLP / GLU / MoE live in ffn.py
    cfg = make_cfg(experts=E, top_k=k, gate=gate)
    blk = ffn.Mixture_of_Experts_SubBlock(cfg).double().eval()   # eval => drop-free
    return cfg, randomize(blk)


@torch.no_grad()
@pytest.mark.parametrize("E,k", [(4, 1), (4, 2), (8, 2)])
@pytest.mark.parametrize("gate", [True, False])
def test_moe_matches_naive_reference_when_drop_free(E, k, gate):
    cfg, blk = make_block(E, k, gate)
    x = torch.randn(3, 10, cfg.embed_dim, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(0))

    y, aux = blk(x)
    torch.testing.assert_close(y, naive_moe(blk, x), atol=ATOL, rtol=0)
    assert aux == 0                                  # aux loss only exists in training


@torch.no_grad()
def test_moe_output_independent_of_batch_composition():
    # Drop-free eval means a token's output must not depend on its batch-mates.
    cfg, blk = make_block(4, 2, True)
    x = torch.randn(4, 8, cfg.embed_dim, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(1))
    # The router runs in float32 by design, and float32 matmuls round differently for a
    # [32, D] batch than for an [8, D] one (~1e-7 relative). So this tolerance is set by
    # float32, not float64. A real bug (a dropped token, a wrong expert) changes the output
    # by O(1), about five orders of magnitude above this threshold.
    torch.testing.assert_close(blk(x)[0][:1], blk(x[:1])[0], atol=1e-5, rtol=1e-5)


def test_moe_training_step_is_finite_and_router_learns():
    cfg = make_cfg(experts=4, top_k=2, cf=1.25)       # capacity < N: some tokens dropped
    model = build_model(cfg).train()
    ids = tokens(16, batch=4)

    logits, aux = model(ids, return_aux_loss=True)
    loss = F.cross_entropy(logits.flatten(0, 1), ids.flatten()) + aux
    loss.backward()

    assert torch.isfinite(loss)
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient in {name}"
        if "router" in name:
            assert p.grad.abs().sum() > 0, f"router {name} receives zero gradient"