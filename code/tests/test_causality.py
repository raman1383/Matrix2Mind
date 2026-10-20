# tests/test_causality.py
#
# TEST 6: the model must be causal.
# Changing token t+1 (and everything after it) must not change the logits at positions <= t.
# A language model that peeks at the future gets a great training loss and then
# cannot generate anything, so this is worth a dedicated test.

import pytest
import torch

from conftest import make_cfg, build_model, tokens

V = 64
T = 32


def perturb_future(ids, t):
    """Change every token after position t (each one is guaranteed to differ)."""
    other = ids.clone()
    other[:, t + 1:] = (other[:, t + 1:] + 7) % V
    return other


def assert_causal(model, t):
    ids = tokens(T, batch=2)
    other = perturb_future(ids, t)
    a, b = model(ids), model(other)

    # the past is untouched ...
    torch.testing.assert_close(a[:, :t + 1], b[:, :t + 1], atol=1e-9, rtol=0)
    # ... and the test is not vacuous: the future really did change
    assert not torch.allclose(a[:, t + 1:], b[:, t + 1:])


CASES = [
    ("dense_mha",          dict(heads=1, kv=1),                                   False),
    ("dense_gqa",          dict(heads=4, kv=2),                                   False),
    ("moe_eval",           dict(heads=4, kv=2, experts=4, top_k=2, cf=None),      False),
    # training mode, capacity_factor=None -> nothing is dropped -> must still be causal
    ("moe_train_no_drops", dict(heads=4, kv=2, experts=4, top_k=2, cf=None),      True),
]


@torch.no_grad()
@pytest.mark.parametrize("t", [0, 5, 20])
@pytest.mark.parametrize("name,kwargs,train", CASES, ids=[c[0] for c in CASES])
def test_future_tokens_do_not_change_past_logits(name, kwargs, train, t):
    model = build_model(make_cfg(window=64, **kwargs))
    model.train(train)
    assert_causal(model, t)


@torch.no_grad()
@pytest.mark.xfail(
    raises=AssertionError,
    strict=False,
    reason="KNOWN LIMITATION (ffn.py, step 2 ASSIGN): capacity is assigned slot-major, so whether "
           "a token's 2nd-choice expert has room depends on the 1st choices of LATER tokens. "
           "Remove this xfail once capacity is assigned token-major / per sequence, or train "
           "with MoE_capacity_factor=None.",
)
def test_moe_capacity_dropping_is_causal_in_training():
    # tiny capacity factor => lots of dropped (token, expert) pairs
    cfg = make_cfg(window=64, heads=4, kv=2, experts=4, top_k=2, cf=0.5)
    model = build_model(cfg).train()
    assert_causal(model, 5)