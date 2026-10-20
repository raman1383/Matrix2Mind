# code/tests/test_kv_cache.py
#
# TEST 1: the ring-buffer KV cache must be invisible.
# Chunked prefill + decode through the cache == one uncached forward pass.

import pytest
import torch

from kv_cache import KV_cache
from conftest import make_cfg, build_model, tokens

ATOL = 1e-9   # float64: real bugs are >> 1e-3, rounding noise is ~1e-13


def run_cached(model, cfg, ids, chunk, num_lanes=1, lane=0):
    cache = KV_cache(cfg, num_lanes=num_lanes, device="cpu", dtype=torch.float64)
    lane_ids = torch.tensor([lane])
    outs = [
        model(ids[:, i:i + chunk], cache=cache, lane_ids=lane_ids)
        for i in range(0, ids.shape[1], chunk)
    ]
    return torch.cat(outs, dim=1)


def window_mask(T, W):
    """query p may attend key a iff p - W < a <= p  (causal + sliding window)"""
    i = torch.arange(T)
    return ((i[None, :] <= i[:, None]) & (i[None, :] > i[:, None] - W))[None, None]


@torch.no_grad()
@pytest.mark.parametrize("heads,kv", [(1, 1), (4, 2)])      # (4, 2) exercises GQA
@pytest.mark.parametrize("chunk", [1, 8, 40])               # pure decode / chunked / one-shot
def test_cached_equals_uncached_within_window(heads, kv, chunk):
    cfg = make_cfg(heads=heads, kv=kv, window=64)
    model = build_model(cfg)
    ids = tokens(40)                                         # 40 <= window: nothing evicted

    torch.testing.assert_close(
        run_cached(model, cfg, ids, chunk), model(ids), atol=ATOL, rtol=0
    )


@torch.no_grad()
@pytest.mark.parametrize("chunk", [1, 7, 16, 24])           # includes chunk > window
def test_cached_equals_windowed_uncached_beyond_window(chunk):
    W, T = 16, 50                                            # T >> W: ring wraps ~3 times
    cfg = make_cfg(window=W)
    model = build_model(cfg)
    ids = tokens(T)

    # Reference: the same model, no cache, but with the sliding-window mask the cache implies.
    expected = model(ids, attn_mask=window_mask(T, W))
    torch.testing.assert_close(
        run_cached(model, cfg, ids, chunk), expected, atol=ATOL, rtol=0
    )


@torch.no_grad()
def test_lanes_are_independent_ragged_and_reusable():
    cfg = make_cfg(window=64)
    model = build_model(cfg)
    a, b, c = tokens(20, seed=1), tokens(11, seed=2), tokens(15, seed=3)
    cache = KV_cache(cfg, num_lanes=2, device="cpu", dtype=torch.float64)

    model(a[:, :-1], cache=cache, lane_ids=torch.tensor([0]))   # lane 0: 19 tokens
    model(b[:, :-1], cache=cache, lane_ids=torch.tensor([1]))   # lane 1: 10 tokens

    # one batched decode step over two lanes of DIFFERENT lengths, in swapped order
    last = torch.cat([b[:, -1:], a[:, -1:]])
    out = model(last, cache=cache, lane_ids=torch.tensor([1, 0]))
    torch.testing.assert_close(out[0, -1], model(b)[0, -1], atol=ATOL, rtol=0)
    torch.testing.assert_close(out[1, -1], model(a)[0, -1], atol=ATOL, rtol=0)

    # free lane 0 and reuse it: stale K/V from `a` must be invisible to `c`
    cache.free(0)
    out = model(c, cache=cache, lane_ids=torch.tensor([0]))
    torch.testing.assert_close(out, model(c), atol=ATOL, rtol=0)


# ---------------------------------------------------------------------------
# host-side length mirror (prepare() must not need a GPU -> CPU sync)
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_host_length_mirror_tracks_device_lengths():
    cfg = make_cfg(window=16)
    model = build_model(cfg)
    cache = KV_cache(cfg, num_lanes=3, device="cpu", dtype=torch.float64)

    model(tokens(5, seed=1), cache=cache, lane_ids=torch.tensor([0]))
    model(tokens(9, seed=2), cache=cache, lane_ids=torch.tensor([2]))
    assert cache._host_len == cache.lengths.tolist() == [5, 0, 9]

    # batched decode step over two lanes, swapped order
    model(tokens(1, seed=3, batch=2), cache=cache, lane_ids=torch.tensor([2, 0]))
    assert cache._host_len == cache.lengths.tolist() == [6, 0, 10]

    cache.free(2)
    assert cache._host_len == cache.lengths.tolist() == [6, 0, 0]


@torch.no_grad()
def test_lane_ids_may_be_a_list_or_a_cpu_tensor():
    cfg = make_cfg(window=16)
    model = build_model(cfg)
    ids = tokens(12)

    outs = []
    for lanes in ([1], torch.tensor([1])):
        cache = KV_cache(cfg, num_lanes=2, device="cpu", dtype=torch.float64)
        outs.append(model(ids, cache=cache, lane_ids=lanes))

    torch.testing.assert_close(outs[0], outs[1], atol=0, rtol=0)
    torch.testing.assert_close(outs[0], model(ids), atol=ATOL, rtol=0)