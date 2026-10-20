# tests/test_data_loader.py
#
# TEST 9: BatchLoader must hand the model int64 ids that match the file exactly.
# The file is uint16. Values above 32767 are the interesting ones: they are the ones a
# signed 16-bit reinterpretation would corrupt, so the source below starts at 60000.

import numpy as np
import pytest
import torch

from data_loader import BatchLoader

SEQ = 8
BATCH = 4

#! passes on Linux and fails on Windows because of memory-mapped files
# def make_loader(tmp_path, n_tokens=200, seed=0):
#     src = np.arange(60000, 60000 + n_tokens, dtype=np.uint16)     # consecutive, all > 32767
#     path = tmp_path / "tokens.bin"
#     src.tofile(path)
#     return src, BatchLoader(path, BATCH, SEQ, "cpu", seed=seed)

def make_loader(tmp_path, n_tokens=200, seed=0):
    src = np.arange(60000, 60000 + n_tokens, dtype=np.uint16)
    path = tmp_path / "tokens.bin"
    if not path.exists():              # a second loader reuses the file the first one has mapped
        src.tofile(path)
    return src, BatchLoader(path, BATCH, SEQ, "cpu", seed=seed)


def test_batches_are_int64_and_y_is_x_shifted_by_one(tmp_path):
    src, loader = make_loader(tmp_path)
    x, y = loader.get_batch()

    assert x.dtype == torch.long and y.dtype == torch.long
    assert x.shape == y.shape == (BATCH, SEQ)
    assert x.min() >= 60000                       # no wrap-around / sign problems
    assert torch.all(y - x == 1)                  # source is consecutive, so y == x + 1
    assert torch.equal(x[:, 1:], y[:, :-1])       # the classic next-token shift


def test_values_come_straight_from_the_file(tmp_path):
    src, loader = make_loader(tmp_path)
    x, y = loader.get_batch()
    for row in range(BATCH):
        start = int(x[row, 0]) - 60000            # consecutive source: value encodes position
        np.testing.assert_array_equal(x[row].numpy(), src[start:start + SEQ].astype(np.int64))
        np.testing.assert_array_equal(y[row].numpy(), src[start + 1:start + SEQ + 1].astype(np.int64))


def test_sequential_batches_cover_distinct_windows_in_order(tmp_path):
    src, loader = make_loader(tmp_path, n_tokens=200)
    batches = list(loader.sequential_batches())

    xs = torch.cat([x for x, _ in batches])
    assert xs.dtype == torch.long
    assert xs.shape[1] == SEQ
    # windows start at 0, SEQ, 2*SEQ, ... : window i is exactly src[i*SEQ : (i+1)*SEQ]
    for i, row in enumerate(xs):
        np.testing.assert_array_equal(row.numpy(), src[i * SEQ:(i + 1) * SEQ].astype(np.int64))

    assert len(list(loader.sequential_batches(max_batches=2))) == 2
    # deterministic: a second pass returns the same tokens
    xs2 = torch.cat([x for x, _ in loader.sequential_batches()])
    assert torch.equal(xs, xs2)


def test_same_seed_same_batches_and_state_roundtrip(tmp_path):
    _, a = make_loader(tmp_path, seed=7)
    _, b = make_loader(tmp_path, seed=7)
    assert torch.equal(a.get_batch()[0], b.get_batch()[0])

    state = a.state_dict()
    expected = a.get_batch()[0]
    b.load_state_dict(state)                      # rewind b to where a was before that draw
    assert torch.equal(b.get_batch()[0], expected)


def test_too_short_file_is_rejected(tmp_path):
    np.arange(SEQ, dtype=np.uint16).tofile(tmp_path / "short.bin")      # needs SEQ + 1 tokens
    with pytest.raises(ValueError):
        BatchLoader(tmp_path / "short.bin", BATCH, SEQ, "cpu")