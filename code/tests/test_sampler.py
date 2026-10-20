# tests/test_sampler.py
#
# TEST 8: sample_next_token_from_logits.
#
# Sampling is random, so we test WHICH tokens can ever be drawn: repeat one logits row
# N times in a batch, draw once per row, and look at the set of tokens that appeared.
# Every probability involved is large enough that a token that is allowed shows up in
# N=4000 draws with probability ~1, so these tests are not flaky.

import math

import pytest
import torch

from inference_engine import sample_next_token_from_logits as sample

N = 4000


def drawn(row, **kwargs) -> set:
    """Set of tokens drawn from `row` (a list of logits) in N independent draws."""
    torch.manual_seed(0)
    logits = torch.tensor(row, dtype=torch.float32).repeat(N, 1)       # [N, V]
    return set(sample(logits, **kwargs).flatten().tolist())


def logits_from_probs(probs):
    return [math.log(p) for p in probs]


# ---------------------------------------------------------------------------
# temperature
# ---------------------------------------------------------------------------

def test_temperature_zero_is_argmax_per_row():
    logits = torch.tensor([[0.1, 2.0, 0.3],
                           [5.0, 0.0, 1.0],
                           [0.0, 0.0, 9.0]])
    out = sample(logits, temperature=0.0)
    assert out.shape == (3, 1)
    assert out.flatten().tolist() == [1, 0, 2]


def test_output_shape_is_batch_by_one():
    out = sample(torch.randn(5, 11), temperature=1.0, top_k=3, top_p=0.9)
    assert out.shape == (5, 1)


def test_very_low_temperature_behaves_like_argmax():
    assert drawn(list(range(10)), temperature=0.01, top_k=None, top_p=None) == {9}


def test_high_temperature_flattens_the_distribution():
    # at T=50 these logits are nearly uniform, so every token shows up;
    # at T=0.1 the 2.0 gap between the top two means the runner-up is ~e^-20 likely: never drawn
    row = [4.0, 3.0, 2.0, 1.0, 0.0]
    assert drawn(row, temperature=50.0, top_k=None, top_p=None) == {0, 1, 2, 3, 4}
    assert drawn(row, temperature=0.1, top_k=None, top_p=None) == {0}


# ---------------------------------------------------------------------------
# top-k
# ---------------------------------------------------------------------------

def test_top_k_keeps_exactly_the_k_best():
    assert drawn(list(range(10)), temperature=1.0, top_k=3, top_p=None) == {7, 8, 9}


def test_top_k_one_is_greedy():
    assert drawn(list(range(10)), temperature=1.0, top_k=1, top_p=None) == {9}


def test_top_k_ties_at_the_threshold_are_all_kept():
    # the 2nd-best logit ties with two others, so "top 2" keeps 3 tokens (documented behaviour)
    assert drawn([1.0, 1.0, 1.0, 0.0, 0.0], temperature=1.0, top_k=2, top_p=None) == {0, 1, 2}


def test_top_k_larger_than_vocab_is_clamped():
    out = drawn(list(range(5)), temperature=1.0, top_k=1000, top_p=None)
    assert out <= {0, 1, 2, 3, 4} and 4 in out


# ---------------------------------------------------------------------------
# top-p
# ---------------------------------------------------------------------------

PROBS = [0.5, 0.3, 0.15, 0.05]       # cumulative: 0.5, 0.8, 0.95, 1.0


@pytest.mark.parametrize("p,expected", [
    (0.40, {0}),                     # token 0 alone already covers 0.5 > 0.4
    (0.70, {0, 1}),                  # 0.5 < 0.7 -> need token 1 too (0.8 > 0.7)
    (0.85, {0, 1, 2}),               # 0.8 < 0.85 -> need token 2 too
    (1.00, {0, 1, 2, 3}),            # p == 1.0 disables the filter
])
def test_top_p_keeps_the_smallest_set_whose_mass_exceeds_p(p, expected):
    assert drawn(logits_from_probs(PROBS), temperature=1.0, top_k=None, top_p=p) == expected


def test_top_p_always_keeps_at_least_one_token():
    assert drawn(logits_from_probs(PROBS), temperature=1.0, top_k=None, top_p=1e-6) == {0}


def test_top_p_is_applied_per_row():
    # row 0 is peaked (one token has 90%), row 1 is flat (four tokens with 25% each)
    torch.manual_seed(0)
    peaked = torch.tensor(logits_from_probs([0.90, 0.04, 0.03, 0.03]))
    flat = torch.tensor(logits_from_probs([0.25, 0.25, 0.25, 0.25]))
    logits = torch.stack([peaked, flat]).repeat_interleave(N, dim=0)       # N rows each
    out = sample(logits, temperature=1.0, top_k=None, top_p=0.5).flatten()
    assert set(out[:N].tolist()) == {0}                                    # nucleus = 1 token
    assert len(set(out[N:].tolist())) >= 2                                 # nucleus = 3 of 4 tokens


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    dict(temperature=-1.0),
    dict(temperature=1.0, top_k=0),
    dict(temperature=1.0, top_k=-3),
    dict(temperature=1.0, top_p=0.0),
    dict(temperature=1.0, top_p=1.5),
])
def test_bad_arguments_raise(kwargs):
    with pytest.raises(ValueError):
        sample(torch.randn(2, 8), **kwargs)


def test_logits_must_be_2d():
    with pytest.raises(ValueError):
        sample(torch.randn(8))
    with pytest.raises(ValueError):
        sample(torch.randn(2, 3, 8))