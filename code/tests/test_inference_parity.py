# tests/test_inference_parity.py
#
# TEST 5: end-to-end greedy parity.
#
# test_kv_cache.py proves the cache is invisible for ONE sequence driven by hand.
# This file proves the whole serving stack (scheduler + chunked prefill + lane reuse +
# ragged batched decode + cache + model) produces exactly the tokens of the naive
# "re-run everything every step" loop. At temperature 0 both must agree token for token.

import dataclasses

import pytest
import torch

from conftest import make_cfg, build_model, randomize, tokens, model_class
from inference_engine import (
    Completion,
    advanced_inference,
    advanced_inference_ids,
    batched_simple_inference,
    batched_simple_inference_ids,
)

W = 16                        # context window
STEPS = 30                    # prompt + generation (up to 50 tokens) wraps the ring ~3 times
PROMPT_LENS = [5, 9, 3, 20]   # ragged; the last prompt is longer than the window itself

GREEDY = dict(temperature=0.0, top_k=None, top_p=None)

ARCHS = [
    ("mha", dict(heads=1, kv=1)),
    ("gqa", dict(heads=4, kv=2)),
    ("moe", dict(heads=4, kv=2, experts=4, top_k=2, cf=None)),   # drop-free
]

# (chunk_size, max_KV_lanes): chunk < window / chunk == window / chunk > window,
# and 1 lane (every request reuses the lane) / 2 lanes (queueing) / 4 lanes (all at once)
SCHEDULES = [(3, 1), (16, 2), (64, 4)]


def make_prompts():
    return [tokens(n, seed=100 + i)[0].tolist() for i, n in enumerate(PROMPT_LENS)]


def reference(model, cfg, prompts, steps):
    """Naive, one prompt at a time (the naive loop needs equal lengths within a batch)."""
    return [
        batched_simple_inference_ids(model, cfg, [p], steps, "cpu", **GREEDY)[0]
        for p in prompts
    ]


# ---------------------------------------------------------------------------
# the main parity test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arch", ARCHS, ids=lambda a: a[0])
@pytest.mark.parametrize("chunk,lanes", SCHEDULES)
def test_greedy_scheduler_matches_naive_reference(arch, chunk, lanes):
    cfg = make_cfg(window=W, **arch[1])
    model = build_model(cfg)
    prompts = make_prompts()

    expected = reference(model, cfg, prompts, STEPS)
    got = advanced_inference_ids(
        model, cfg, prompts, "cpu",
        max_new_tokens=STEPS, max_KV_lanes=lanes, chunk_size=chunk, **GREEDY,
    )

    for i, (g, e) in enumerate(zip(got, expected)):
        assert g.finish_reason == "length"
        assert g.generated == e, f"prompt {i} (len {PROMPT_LENS[i]}) diverged from the naive loop"


def test_results_come_back_in_input_order():
    # 4 prompts / 2 lanes: requests finish in a different order than they were submitted
    cfg = make_cfg(window=W)
    model = build_model(cfg)
    prompts = make_prompts()

    got = advanced_inference_ids(model, cfg, prompts, "cpu", max_new_tokens=8,
                                 max_KV_lanes=2, chunk_size=4, **GREEDY)
    assert [g.generated for g in got] == reference(model, cfg, prompts, 8)


# ---------------------------------------------------------------------------
# stopping rules
# ---------------------------------------------------------------------------

def test_eos_stops_generation_and_is_stripped():
    cfg = make_cfg(window=W)
    model = build_model(cfg)
    prompts = make_prompts()[:2]
    full = reference(model, cfg, prompts, STEPS)

    eos = full[0][6]                      # a token we KNOW the model emits for prompt 0
    got = advanced_inference_ids(model, cfg, prompts, "cpu", max_new_tokens=STEPS,
                                 max_KV_lanes=2, chunk_size=4, eos_token_id=eos, **GREEDY)

    k0 = full[0].index(eos)               # first occurrence
    assert got[0].finish_reason == "eos"
    assert got[0].generated == full[0][:k0]            # EOS itself is not returned

    # prompt 1 stops at its own first EOS, or runs to the length limit if it never emits it
    if eos in full[1]:
        k1 = full[1].index(eos)
        assert (got[1].finish_reason, got[1].generated) == ("eos", full[1][:k1])
    else:
        assert (got[1].finish_reason, got[1].generated) == ("length", full[1])


@pytest.mark.parametrize("n", [0, 1, 2])
def test_max_new_tokens_is_exact(n):
    cfg = make_cfg(window=W)
    model = build_model(cfg)
    prompts = make_prompts()

    got = advanced_inference_ids(model, cfg, prompts, "cpu", max_new_tokens=n,
                                 max_KV_lanes=2, chunk_size=4, **GREEDY)

    assert [len(g.generated) for g in got] == [n] * len(prompts)
    assert all(g.finish_reason == "length" for g in got)
    assert [g.generated for g in got] == reference(model, cfg, prompts, n)


def test_zero_new_tokens_never_runs_the_model():
    cfg = make_cfg(window=W)
    model = build_model(cfg)
    calls = []
    model.register_forward_pre_hook(lambda module, args: calls.append(1))

    advanced_inference_ids(model, cfg, make_prompts(), "cpu", max_new_tokens=0, **GREEDY)
    batched_simple_inference_ids(model, cfg, [make_prompts()[0]], 0, "cpu", **GREEDY)

    assert calls == []


def test_invalid_arguments_are_rejected():
    cfg = make_cfg(window=W)
    model = build_model(cfg)
    p = make_prompts()

    with pytest.raises(ValueError):
        advanced_inference_ids(model, cfg, p, "cpu", max_new_tokens=-1)
    with pytest.raises(ValueError):
        advanced_inference_ids(model, cfg, p, "cpu", chunk_size=0)      # would loop forever
    with pytest.raises(ValueError):
        advanced_inference_ids(model, cfg, p, "cpu", max_KV_lanes=0)
    with pytest.raises(ValueError):
        advanced_inference_ids(model, cfg, [[]], "cpu")                 # empty prompt
    with pytest.raises(ValueError):
        batched_simple_inference_ids(model, cfg, p, 3, "cpu")           # unequal lengths
    with pytest.raises(ValueError):
        batched_simple_inference_ids(model, cfg, [p[0]], -1, "cpu")


# ---------------------------------------------------------------------------
# the naive baseline itself
# ---------------------------------------------------------------------------

def test_naive_baseline_is_batch_invariant_and_handles_the_window():
    cfg = make_cfg(window=W)
    model = build_model(cfg)
    prompts = [tokens(10, seed=s)[0].tolist() for s in (1, 2, 3)]   # equal lengths

    # 10 + 12 = 22 > W: the last steps use the sliding-window mask
    batched = batched_simple_inference_ids(model, cfg, prompts, 12, "cpu", **GREEDY)
    single = [batched_simple_inference_ids(model, cfg, [p], 12, "cpu", **GREEDY)[0] for p in prompts]
    assert batched == single


# ---------------------------------------------------------------------------
# string wrappers (tokenizer in the loop)
# ---------------------------------------------------------------------------

def test_string_api_smoke():
    cfg = dataclasses.replace(make_cfg(window=32), vocab_size=2048)   # real tokenizer vocab size
    model = randomize(model_class()(cfg).double(), std=0.3).eval()
    prompts = ["once upon a time", "the little dog"]

    out = advanced_inference(model, cfg, prompts, "cpu", max_new_tokens=6, **GREEDY)

    assert all(isinstance(c, Completion) for c in out)
    for prompt, c in zip(prompts, out):
        assert c.text == prompt + c.completion
        assert c.finish_reason == "length"

    naive = batched_simple_inference(model, cfg, 6, [prompts[0]], "cpu", **GREEDY)
    assert naive[0] == out[0].text