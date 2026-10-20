"""
Compare training runs in ONE image.

Every sub-folder of ../runs that contains a metrics.jsonl is one run, and the sub-folder's
name is the label (metrics.jsonl itself does not store the model name).

    python plot_runs.py                                  # every run folder in ../runs
    python plot_runs.py --runs dense_8D MoE_top2         # only these folders, in this order
    python plot_runs.py --exclude smoke scale_up_1K      # skip folders whose name contains any of these
    python plot_runs.py --out ../runs/compare.png --style light

The figure is a 2x2 grid that shares one colour per run:
    top row     the whole run:    training cross-entropy (average over each logging window,
                                  `train_ce` records) | validation loss (`val_loss` records)
    bottom row  the same two curves zoomed on the second half of the run (--zoom-from),
                                  y-axis fitted to the data, so small differences are visible
The legend shows each run's FINAL validation loss: the full-validation-set number
(`val_loss_full`) when the run has finished, otherwise the last evaluation, marked as unfinished.

Only needs matplotlib. Default runs folder: ../runs next to the code/ folder, wherever you launch from.
"""
import argparse
import json
import math
import sys
from pathlib import Path

DEFAULT_RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"

# colour-blind-safe (Okabe-Ito based); one palette per background so no colour disappears
PALETTES = {
    "dark":  ["#56B4E9", "#E69F00", "#009E73", "#CC79A7", "#F0E442", "#D55E00", "#FFFFFF", "#8DA0CB"],
    "light": ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#7A7A7A"],
}
LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]   # second cue besides colour


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass                                        # torn last line from a crash
    return out


def load_run(run_dir: Path):
    """metrics.jsonl of one run folder -> dict, or None if there is nothing to plot."""
    path = run_dir / "metrics.jsonl"
    if not path.is_file():
        return None

    train, val, full = {}, {}, None
    for r in read_jsonl(path):
        step = r.get("step")
        if step is None:
            continue
        if "train_ce" in r:
            train[step] = r["train_ce"]
        if "val_loss" in r:
            val[step] = r["val_loss"]
        if "val_loss_full" in r:
            full = r["val_loss_full"]
    if not train and not val:
        return None

    train, val = sorted(train.items()), sorted(val.items())
    if full is not None:
        final, finished = full, True
    elif val:
        final, finished = val[-1][1], False
    else:
        final, finished = float("nan"), False
    return dict(
        name=run_dir.name, train=train, val=val, final=final, finished=finished,
        last_step=max([s for s, _ in train] + [s for s, _ in val]),
        best_val=min((v for _, v in val), default=float("nan")),
    )


def discover(runs_dir: Path, wanted, excluded):
    if not runs_dir.is_dir():
        sys.exit(f"runs folder not found: {runs_dir}")

    available = sorted(p for p in runs_dir.iterdir() if p.is_dir())
    if wanted:
        by_name = {p.name: p for p in available}
        missing = [n for n in wanted if n not in by_name]
        if missing:
            sys.exit(f"no such run folder(s) in {runs_dir}: {missing}\navailable: {sorted(by_name)}")
        folders = [by_name[n] for n in wanted]
    else:
        folders = available
    folders = [p for p in folders if not any(x in p.name for x in excluded)]

    runs, skipped = [], []
    for p in folders:
        run = load_run(p)
        (runs if run else skipped).append(run or p.name)
    return runs, skipped


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def make_figure(runs, style, zoom_from, vocab_size, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = PALETTES[style]
    with plt.style.context("dark_background" if style == "dark" else "default"):
        fig, axes = plt.subplots(2, 2, figsize=(14, 9.6))
        (ax_train, ax_val), (ax_train_z, ax_val_z) = axes
        grid = dict(color="#444444" if style == "dark" else "#cccccc", alpha=0.7, linewidth=0.6)

        max_step = max(r["last_step"] for r in runs)
        zoom_step = zoom_from * max_step
        train_zoom_vals, val_zoom_vals = [], []
        handles = []

        for i, r in enumerate(runs):
            color, ls = palette[i % len(palette)], LINESTYLES[i % len(LINESTYLES)]
            kw = dict(color=color, linestyle=ls, linewidth=1.9)

            if r["train"]:
                ax_train.plot(*zip(*r["train"]), **kw)
                tz = [(x, y) for x, y in r["train"] if x >= zoom_step]
                if tz:
                    ax_train_z.plot(*zip(*tz), **kw)
                    train_zoom_vals += [y for _, y in tz]
            if r["val"]:
                xs, ys = zip(*r["val"])
                line, = ax_val.plot(xs, ys, marker="o", markersize=2.8, **kw)
                vz = [(x, y) for x, y in r["val"] if x >= zoom_step]
                if vz:
                    ax_val_z.plot(*zip(*vz), marker="o", markersize=3.5, **kw)
                    val_zoom_vals += [y for _, y in vz]
            else:
                line, = ax_val.plot([], [], **kw)           # keeps a legend entry for a run with no evals

            tag = "" if r["finished"] else "  (unfinished, last eval)"
            handles.append((line, f"{r['name']}  -  final val {r['final']:.4f}{tag}"))

        if vocab_size:
            rnd = math.log(vocab_size)
            base = dict(color="#ff5555" if style == "dark" else "#cc0000", linestyle=(0, (4, 3)), linewidth=1.1, alpha=0.9)
            for ax in (ax_train, ax_val):
                ax.axhline(rnd, **base)
            rnd_line = ax_val.axhline(rnd, **base)
            handles.append((rnd_line, f"random guess  ln({vocab_size}) = {rnd:.3f}"))

        ax_train.set_title("Training loss (cross-entropy, window average)")
        ax_val.set_title("Validation loss")
        ax_train_z.set_title(f"Training loss, zoom (step >= {zoom_step:,.0f})")
        ax_val_z.set_title(f"Validation loss, zoom (step >= {zoom_step:,.0f})")
        for ax in axes.flat:
            ax.set_xlabel("step")
            ax.grid(True, **grid)
        for ax in (ax_train, ax_train_z):
            ax.set_ylabel("loss (nats / token)")

        for ax, vals in ((ax_train_z, train_zoom_vals), (ax_val_z, val_zoom_vals)):
            if vals:
                lo, hi = min(vals), max(vals)
                pad = max(0.05 * (hi - lo), 1e-3)
                ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlim(left=zoom_step)

        fig.suptitle(title or f"Run comparison  ({len(runs)} runs)", fontsize=15, y=0.995)
        fig.legend([h for h, _ in handles], [t for _, t in handles], loc="lower center",
                   ncol=2 if len(handles) > 3 else 1, frameon=False, fontsize=11)
        legend_rows = math.ceil(len(handles) / (2 if len(handles) > 3 else 1))
        fig.tight_layout(rect=[0, 0.03 * legend_rows + 0.015, 1, 0.965])
    return fig


def print_table(runs):
    w = max(len(r["name"]) for r in runs)
    print(f"\n{'run':<{w}}  {'last step':>9}  {'final val':>9}  {'best eval':>9}  status")
    for r in runs:
        print(f"{r['name']:<{w}}  {r['last_step']:>9,}  {r['final']:>9.4f}  {r['best_val']:>9.4f}  "
              f"{'finished' if r['finished'] else 'UNFINISHED'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR, help=f"default: {DEFAULT_RUNS_DIR}")
    ap.add_argument("--runs", nargs="+", help="only these run folders, in this order (default: all)")
    ap.add_argument("--exclude", nargs="+", default=[], metavar="TEXT", help="skip folders whose name contains TEXT")
    ap.add_argument("--out", type=Path, default=None, help="default: <runs-dir>/loss_comparison.png")
    ap.add_argument("--style", choices=["dark", "light"], default="dark")
    ap.add_argument("--zoom-from", type=float, default=0.5, help="zoom panel starts at this fraction of the run (default 0.5)")
    ap.add_argument("--vocab", type=int, default=2048, help="draws the random-guess line ln(vocab); 0 hides it")
    ap.add_argument("--dpi", type=int, default=170)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    runs, skipped = discover(args.runs_dir, args.runs, args.exclude)
    if skipped:
        print(f"skipped (no usable metrics.jsonl): {skipped}")
    if not runs:
        sys.exit(f"no runs with a metrics.jsonl found in {args.runs_dir}")
    print(f"plotting {len(runs)} run(s): {[r['name'] for r in runs]}")
    if len(runs) > len(PALETTES[args.style]):
        print("warning: more runs than distinct colours; colours repeat (line styles still differ)")

    fig = make_figure(runs, args.style, args.zoom_from, args.vocab, args.title)
    out = args.out or (args.runs_dir / "loss_comparison.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, facecolor=fig.get_facecolor())
    print_table(runs)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()