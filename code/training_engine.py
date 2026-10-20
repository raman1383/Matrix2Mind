import json
import math
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from einops import rearrange

from data_loader import BatchLoader
from model_config import ModelConfig, DataConfig, TrainConfig


CHECKPOINT_VERSION = 2

# TrainConfig fields that are baked into optimizer / scheduler state. Changing them
# on resume would be silently ignored (the saved state wins), so we refuse instead.
# Everything else in TrainConfig (eval/log/checkpoint settings, grad_clip, amp,
# seed...) may differ between stages.
_TRAIN_MUST_MATCH = (
    "lr", "total_steps", "warmup_steps", "min_lr_ratio",
    "weight_decay", "betas", "batch_size",
)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _cross_entropy(logits, targets, reduction="mean"):
    return nn.functional.cross_entropy(
        rearrange(logits, "b t v -> (b t) v"),
        rearrange(targets, "b t -> (b t)"),
        reduction=reduction,
    )


def _unwrap(model: nn.Module) -> nn.Module:
    """torch.compile wraps the model; its state_dict keys get an '_orig_mod.' prefix."""
    return getattr(model, "_orig_mod", model)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Optimizer + LR schedule
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, cfg: TrainConfig, device_type: str):
    """AdamW with weight decay on matrices only (not norms / biases / embeddings)."""
    decay, no_decay = [], []
    for name, p in _unwrap(model).named_parameters():
        if not p.requires_grad:
            continue
        leaf = name.split(".")[-1]
        if p.ndim < 2 or leaf in ("embed", "unembed"):
            no_decay.append(p)
        else:
            decay.append(p)

    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    groups = [g for g in groups if g["params"]]

    return torch.optim.AdamW(
        groups, lr=cfg.lr, betas=cfg.betas, eps=1e-8,
        fused=(device_type == "cuda"),
    )


def build_lr_scheduler(optimizer, total_steps: int, warmup_steps: int = 0,
                       min_lr_ratio: float = 0.1):
    """
    Linear warmup -> cosine decay to (min_lr_ratio * peak LR).
    Pure function of the step index, so resuming is exact.
    Constant LR == warmup_steps=0, min_lr_ratio=1.0.
    """

    def lr_factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)


# ---------------------------------------------------------------------------
# Metrics log (JSONL = source of truth for curves)
# ---------------------------------------------------------------------------

class MetricsLogger:
    """
    Append-only JSONL. `keep_through=None` starts a fresh file; `keep_through=N`
    (resume) first drops records with step > N, so a crash between checkpoints
    never leaves duplicate steps in the log.
    """

    def __init__(self, path: Path, keep_through: int | None = None):
        self.path = Path(path)
        records = []
        if keep_through is not None and self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue                      # torn last line from a crash
                    if rec.get("step", 0) <= keep_through:
                        records.append(rec)
        with open(self.path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        self.f = open(self.path, "a", encoding="utf-8")

    def log(self, step: int, **kv):
        self.f.write(json.dumps({"step": step, **kv}) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def read_metrics(run_dir) -> list[dict]:
    path = Path(run_dir) / "metrics.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Checkpointing (everything lives in the run directory)
# ---------------------------------------------------------------------------

def _atomic_torch_save(obj, path: Path) -> None:
    """Write to a temp file, then rename (os.replace is atomic)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _checkpoint_path(run_dir: Path, step: int) -> Path:
    return run_dir / f"ckpt_step{step:08d}.pt"       # zero-padded: sort == step order


def list_checkpoints(run_dir) -> list[Path]:
    return sorted(Path(run_dir).glob("ckpt_step*.pt"))


def find_latest_checkpoint(run_dir) -> Path | None:
    ckpts = list_checkpoints(run_dir)
    return ckpts[-1] if ckpts else None


def _rotate_checkpoints(run_dir: Path, keep: int) -> None:
    if keep is None or keep <= 0:
        return
    for old in list_checkpoints(run_dir)[:-keep]:
        old.unlink()


def save_checkpoint(
    run_dir: Path,
    step: int,                       # number of COMPLETED steps (= next step to run)
    model: nn.Module,
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    optimizer,
    scheduler,
    scaler,
    train_loader: BatchLoader,
    stats: torch.Tensor,             # running [ce_sum, aux_sum, gnorm_sum] since last log
    steps_since_log: int,
) -> Path:
    state = {
        "version": CHECKPOINT_VERSION,
        "step": step,

        # provenance
        "model_config": asdict(model_cfg),
        "data_config": asdict(data_cfg),
        "train_config": asdict(train_cfg),
        "git_commit": _git_commit(),
        "torch_version": str(torch.__version__),   # TorchVersion is a str subclass; weights_only=True rejects it

        # state
        "model": _unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "train_loader": train_loader.state_dict(),
        "rng": {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        },

        # log accumulators, so the first log line after a resume is exact
        "stats": stats.detach().cpu(),
        "steps_since_log": int(steps_since_log),
    }

    path = _checkpoint_path(run_dir, step)
    _atomic_torch_save(state, path)
    _rotate_checkpoints(run_dir, train_cfg.keep_last_n)
    print(f"Checkpoint saved: {path}")
    return path


def _diff(saved: dict, current: dict, label: str, keys=None) -> list[str]:
    keys = current.keys() if keys is None else keys
    return [
        f"  {label}.{k}: checkpoint={saved.get(k)!r} vs current={current[k]!r}"
        for k in keys
        if saved.get(k) != current[k]
    ]


def load_checkpoint(
    path,
    model: nn.Module,
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    optimizer,
    scheduler,
    scaler,
    train_loader: BatchLoader,
) -> dict:
    """Restore model / optimizer / scheduler / loader / RNG state in place."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location="cpu", weights_only=True)

    if ckpt.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version {ckpt.get('version')} "
            f"(expected {CHECKPOINT_VERSION}). Checkpoints written before the "
            f"config split are not loadable; retrain."
        )

    problems = (
        _diff(ckpt["model_config"], asdict(model_cfg), "model")
        + _diff(ckpt["data_config"], asdict(data_cfg), "data")
        + _diff(ckpt["train_config"], asdict(train_cfg), "train", _TRAIN_MUST_MATCH)
    )
    if problems:
        raise ValueError("Config differs from checkpoint:\n" + "\n".join(problems))

    _unwrap(model).load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    if ckpt["scaler"]:                       # {} when the saved run had no GradScaler
        scaler.load_state_dict(ckpt["scaler"])

    train_loader.load_state_dict(ckpt["train_loader"])

    torch.set_rng_state(ckpt["rng"]["torch_cpu"])
    cuda_states = ckpt["rng"]["torch_cuda"]
    if (
        cuda_states is not None
        and torch.cuda.is_available()
        and len(cuda_states) == torch.cuda.device_count()
    ):
        torch.cuda.set_rng_state_all(cuda_states)

    print(f"Resumed from {path} at step {ckpt['step']}")
    return ckpt


def _resolve_resume_path(run_dir: Path, resume_from) -> Path | None:
    if resume_from is None:
        return None
    if resume_from == "latest":
        latest = find_latest_checkpoint(run_dir)
        if latest is None:
            print(f"No checkpoints in {run_dir}; starting from scratch.")
        return latest
    return Path(resume_from)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: BatchLoader, amp_dtype=None,
             max_batches: int | None = None):
    """
    Mean per-token cross-entropy over a deterministic validation pass.
    Same `max_batches` => same tokens => comparable points on the curve.
    Returns (mean_loss_nats, perplexity).
    """
    was_training = model.training
    model.eval()
    total, n_tokens = None, 0
    try:
        for x, y in loader.sequential_batches(max_batches):
            with torch.autocast(x.device.type, dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                logits = model(x)
            loss = _cross_entropy(logits.float(), y, reduction="sum")
            total = loss if total is None else total + loss
            n_tokens += y.numel()
    finally:
        model.train(was_training)

    if total is None or n_tokens == 0:
        raise ValueError("Validation loader produced no tokens")

    mean_loss = (total / n_tokens).item()
    return mean_loss, math.exp(mean_loss)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    data_cfg: DataConfig,
    device,
    resume_from: str | Path | None = None,   # None | "latest" | path to .pt
    stop_at_step: int | None = None,         # run only part of the schedule
) -> Path:
    """
    Train up to `stop_at_step` (default: train_cfg.total_steps). The LR schedule always
    spans train_cfg.total_steps, so staged runs (stop_at_step=1000, then 2000, ...)
    follow exactly the same curve as one uninterrupted run.

    Writes into train_cfg.run_dir:
        metrics.jsonl            train/val metrics
        ckpt_step########.pt     resumable checkpoints (last `keep_last_n`)
        final.pt                 weights only, written when total_steps is reached
    Returns the run directory.
    """
    device = torch.device(device)
    run_dir = train_cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    total = train_cfg.total_steps
    end_step = total if stop_at_step is None else min(stop_at_step, total)

    is_moe = model_cfg.ffn is not None and model_cfg.ffn.MoE_num_experts > 1
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(train_cfg.amp)
    tokens_per_step = train_cfg.batch_size * model_cfg.max_context_window

    model = model.to(device)
    optimizer = build_optimizer(model, train_cfg, device.type)
    scheduler = build_lr_scheduler(
        optimizer, total, train_cfg.warmup_steps, train_cfg.min_lr_ratio
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(amp_dtype == torch.float16 and device.type == "cuda")
    )

    train_loader = BatchLoader(
        data_cfg.train_path, train_cfg.batch_size, model_cfg.max_context_window,
        device, seed=train_cfg.seed,
    )
    val_loader = BatchLoader(
        data_cfg.val_path, train_cfg.batch_size, model_cfg.max_context_window, device,
    )

    stats = torch.zeros(3, device=device)    # [ce_sum, aux_sum, grad_norm_sum]
    steps_since_log = 0
    start_step = 0

    # ---- resume -----------------------------------------------------------
    ckpt_path = _resolve_resume_path(run_dir, resume_from)
    if ckpt_path is not None:
        ckpt = load_checkpoint(
            ckpt_path, model, model_cfg, data_cfg, train_cfg,
            optimizer, scheduler, scaler, train_loader,
        )
        start_step = ckpt["step"]
        stats.copy_(ckpt["stats"].to(device))
        steps_since_log = ckpt["steps_since_log"]

        if start_step >= end_step:
            print(f"Checkpoint is at step {start_step} >= {end_step}. Nothing to do.")
            return run_dir
    elif list_checkpoints(run_dir):
        raise ValueError(
            f"{run_dir} already contains checkpoints. Pass resume_from='latest' to "
            f"continue, or choose a new run_name / delete the directory."
        )

    logger = MetricsLogger(
        run_dir / "metrics.jsonl",
        keep_through=start_step if ckpt_path is not None else None,
    )

    model.train()

    # Sanity point: at step 0 the loss should be ~ln(vocab_size).
    if start_step == 0:
        val, ppl = evaluate(model, val_loader, amp_dtype, train_cfg.eval_batches)
        logger.log(0, val_loss=val, val_ppl=ppl)
        print(f"Step {0:05d} | val {val:.4f} | val ppl {ppl:.2f}")

    last_train_ce, last_gnorm = None, None
    t_last, step_at_t_last = time.perf_counter(), start_step

    # ---- main loop --------------------------------------------------------
    for step in range(start_step, end_step):
        x, y = train_loader.get_batch()

        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            if is_moe:
                logits, aux = model(x, return_aux_loss=True)
            else:
                logits = model(x)
                aux = torch.zeros((), device=device)
            ce = _cross_entropy(logits.float(), y)
        loss = ce + aux

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)                 # no-op when the scaler is disabled
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Accumulate CE (not CE+aux) so the train curve is comparable to val loss.
        # Everything stays on device: no .item() per step.
        stats += torch.stack([ce.detach(), aux.detach().float(), gnorm.detach().float()])
        steps_since_log += 1
        completed = step + 1

        if completed % train_cfg.log_every == 0 or completed == end_step:
            ce_m, aux_m, gn_m = (stats / steps_since_log).tolist()   # syncs the GPU
            stats.zero_()
            steps_since_log = 0
            now = time.perf_counter()
            tok_s = (completed - step_at_t_last) * tokens_per_step / (now - t_last)
            t_last, step_at_t_last = now, completed
            last_train_ce, last_gnorm = ce_m, gn_m
            logger.log(
                completed, train_ce=ce_m, aux=aux_m, grad_norm=gn_m,
                lr=optimizer.param_groups[0]["lr"],
                tokens_seen=completed * tokens_per_step, tok_per_s=tok_s,
            )

        if completed % train_cfg.eval_every == 0 or completed == end_step:
            val, ppl = evaluate(model, val_loader, amp_dtype, train_cfg.eval_batches)
            logger.log(completed, val_loss=val, val_ppl=ppl)
            tr = f"{last_train_ce:.4f}" if last_train_ce is not None else "n/a"
            gn = f"{last_gnorm:.2f}" if last_gnorm is not None else "n/a"
            print(
                f"Step {completed:05d} | train(avg) {tr} | val {val:.4f} | "
                f"val ppl {ppl:.2f} | lr {optimizer.param_groups[0]['lr']:.2e} | "
                f"gnorm {gn}"
            )

        if (
            train_cfg.checkpoint_every
            and completed % train_cfg.checkpoint_every == 0
            and completed != end_step          # the end-of-stage one is written below
        ):
            save_checkpoint(
                run_dir, completed, model, model_cfg, data_cfg, train_cfg,
                optimizer, scheduler, scaler, train_loader, stats, steps_since_log,
            )

    # ---- finish -----------------------------------------------------------
    save_checkpoint(
        run_dir, end_step, model, model_cfg, data_cfg, train_cfg,
        optimizer, scheduler, scaler, train_loader, stats, steps_since_log,
    )

    if end_step == total:
        full, full_ppl = evaluate(model, val_loader, amp_dtype)      # whole val set, once
        logger.log(total, val_loss_full=full, val_ppl_full=full_ppl)
        _atomic_torch_save(_unwrap(model).state_dict(), run_dir / "final.pt")
        print(f"\nFull validation: loss {full:.4f} | ppl {full_ppl:.2f}")
        print(f"Weights saved to {run_dir / 'final.pt'}")

    logger.close()
    return run_dir


# ---------------------------------------------------------------------------
# Plots / inspection
# ---------------------------------------------------------------------------

def plot_loss_history(run_dir, model_cfg: ModelConfig):
    records = read_metrics(run_dir)
    val = [(r["step"], r["val_loss"]) for r in records if "val_loss" in r]
    trn = [(r["step"], r["train_ce"]) for r in records if "train_ce" in r]
    random_loss = math.log(model_cfg.vocab_size)

    plt.style.use("dark_background")
    plt.figure(figsize=(10, 5))
    if trn:
        plt.plot(*zip(*trn), color="#888888", alpha=0.8, label="train CE (avg)")
    plt.plot(*zip(*val), color="#00FFCC", label="val loss")
    plt.axhline(random_loss, color="red", linestyle="--",
                label=f"random ln({model_cfg.vocab_size}) ≈ {random_loss:.3f}")
    plt.title(f"Loss ({Path(run_dir).name})")
    plt.xlabel("Step")
    plt.ylabel("Cross-Entropy (nats)")
    plt.grid(True, color="#333333")
    plt.legend()
    plt.show()



def inspect_weight_file(weight_file_path, print_weights: bool = False):
    weight_file_path = Path(weight_file_path)
    if not weight_file_path.exists():
        raise FileNotFoundError(f"Weight file not found: {weight_file_path}")

    state_dict = torch.load(weight_file_path, map_location="cpu", weights_only=True)
    print(f"Loaded state_dict from {weight_file_path}")
    print(f"Number of parameter tensors: {len(state_dict)}")
    print(f"Number of parameters: {sum(p.numel() for p in state_dict.values()):,}")

    print("\n" + "=" * 80 + "\nPARAMETER SUMMARY\n" + "=" * 80)
    for name, param in state_dict.items():
        print(f"{name}: shape={tuple(param.shape)}, dtype={param.dtype}, "
              f"numel={param.numel():,}")

    if print_weights:
        print("\n" + "=" * 80 + "\nPARAMETER TENSORS\n" + "=" * 80)
        for name, param in state_dict.items():
            print(f"\n{name}\n" + "-" * 80)
            print(param)