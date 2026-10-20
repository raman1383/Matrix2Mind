# tests/test_resume.py
#
# TEST 3: stopping and resuming must be indistinguishable from never stopping.
# Self-contained: uses a tiny model defined here, synthetic data, CPU only
# (GPU kernels are not bit-deterministic, CPU ones are).

import dataclasses

import numpy as np
import pytest
import torch
import torch.nn as nn

import training_engine as te
from model_config import ModelConfig, DataConfig, TrainConfig


class TinyLM(nn.Module):
    """Has a 2-D non-embedding matrix (-> weight-decay group) and embeddings (-> no-decay)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        D, V = cfg.embed_dim, cfg.vocab_size
        self.embed = nn.Parameter(torch.randn(V, D) * 0.1)
        self.proj = nn.Parameter(torch.randn(D, D) * 0.1)
        self.unembed = nn.Parameter(torch.randn(D, V) * 0.1)

    def forward(self, idx):
        return (self.embed[idx] @ self.proj) @ self.unembed


def _fresh_model(cfg, seed):
    torch.manual_seed(seed)
    return TinyLM(cfg)


def _write_tokens(path, n, vocab, seed):
    np.random.default_rng(seed).integers(0, vocab, n).astype(np.uint16).tofile(path)


@pytest.fixture
def setup(tmp_path):
    model_cfg = ModelConfig(embed_dim=16, vocab_size=64, max_context_window=8)
    _write_tokens(tmp_path / "train.bin", 5000, 64, seed=0)
    _write_tokens(tmp_path / "val.bin", 1000, 64, seed=1)
    data_cfg = DataConfig(str(tmp_path / "train.bin"), str(tmp_path / "val.bin"))
    train_cfg = TrainConfig(
        run_name="x", run_root=str(tmp_path / "runs"),
        total_steps=40, batch_size=4, lr=1e-2, warmup_steps=5, weight_decay=0.1,
        eval_every=10, eval_batches=3, log_every=5, checkpoint_every=10,
    )
    return model_cfg, data_cfg, train_cfg


def _stable(run_dir):
    # everything in the log except wall-clock throughput
    return [{k: v for k, v in r.items() if k != "tok_per_s"}
            for r in te.read_metrics(run_dir)]


def test_resume_is_bit_exact(setup):
    model_cfg, data_cfg, cfg = setup
    cpu = torch.device("cpu")

    # A: 40 steps, uninterrupted
    run_a = te.train(_fresh_model(model_cfg, seed=0), model_cfg,
                     dataclasses.replace(cfg, run_name="straight"), data_cfg, cpu)

    # B: 20 steps, "crash", brand-new process state, resume to 40.
    # The stage-2 model starts from DIFFERENT random weights (seed=123), so the
    # test only passes if everything really comes back from the checkpoint.
    b = dataclasses.replace(cfg, run_name="staged")
    te.train(_fresh_model(model_cfg, seed=0), model_cfg, b, data_cfg, cpu, stop_at_step=20)
    run_b = te.train(_fresh_model(model_cfg, seed=123), model_cfg, b, data_cfg, cpu,
                     resume_from="latest")

    # 1) final weights identical to the last bit
    wa = torch.load(run_a / "final.pt", weights_only=True)
    wb = torch.load(run_b / "final.pt", weights_only=True)
    assert wa.keys() == wb.keys()
    for k in wa:
        assert torch.equal(wa[k], wb[k]), f"weights differ after resume: {k}"

    # 2) identical training curves: same data order, LR schedule, grad norms, evals,
    #    and no duplicated or missing records around the resume point
    assert _stable(run_a) == _stable(run_b)