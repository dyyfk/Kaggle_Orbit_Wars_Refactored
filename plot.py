"""Render training_curves.png from a JSONL log written by train.py.

Usage:
    python plot.py [log_path] [--out training_curves.png]

The log has three row types (one JSON object per line):
  - meta:  recorded at start; contains argparse args + start time
  - step:  one per training step; loss, batch win/total, return_mean, elapsed
  - eval:  one per eval (including baseline at step=0); win, win_ci, margin,
           flip_rate, mean_max_delta, n

The output is a 2x2 PNG: loss, win rate (train scatter + eval line with 95%
CI band), eval margin, MLP diagnostics (flip rate + mean |delta|max).
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_log(path: Path):
    steps, evals, meta = [], [], None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            t = row.get("type")
            if t == "step":
                steps.append(row)
            elif t == "eval":
                evals.append(row)
            elif t == "meta":
                meta = row
    return meta, steps, evals


def ema(xs, span=10):
    if not xs:
        return []
    alpha = 2.0 / (span + 1)
    out = [xs[0]]
    for x in xs[1:]:
        out.append(alpha * x + (1 - alpha) * out[-1])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log_path", nargs="?", default="training_log.jsonl")
    p.add_argument("--out", default="training_curves.png")
    args = p.parse_args()

    log_path = Path(args.log_path)
    out_path = Path(args.out)
    if not log_path.exists():
        print(f"Log not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    meta, steps, evals = parse_log(log_path)
    if not steps and not evals:
        print(f"No step/eval rows in {log_path}", file=sys.stderr)
        sys.exit(1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Training log: {log_path.name}  "
                 f"(steps={len(steps)}, evals={len(evals)})", fontsize=11)

    # ----- (0,0) Loss --------------------------------------------------------
    ax = axes[0, 0]
    if steps:
        xs = [s["step"] for s in steps]
        ys = [s["loss"] for s in steps]
        ax.plot(xs, ys, color="#aaa", alpha=0.6, lw=1, label="loss (raw)")
        ax.plot(xs, ema(ys, span=10), color="C0", lw=2, label="EMA(10)")
    ax.axhline(0, color="black", lw=0.5, alpha=0.3)
    ax.set_xlabel("step")
    ax.set_ylabel("PG loss")
    ax.set_title("Policy gradient loss")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    # ----- (0,1) Win rate ----------------------------------------------------
    ax = axes[0, 1]
    if steps:
        xs = [s["step"] for s in steps]
        ys = [s["batch_win"] / max(1, s["batch_total"]) for s in steps]
        ax.plot(xs, ys, "o", color="#bbb", ms=3, alpha=0.5,
                label="train batch")
    if evals:
        ex = [e["step"] for e in evals]
        ey = [e["win"] for e in evals]
        ec = [e["win_ci"] for e in evals]
        ax.fill_between(ex,
                        [y - c for y, c in zip(ey, ec)],
                        [y + c for y, c in zip(ey, ec)],
                        color="C0", alpha=0.2, label="eval 95% CI")
        ax.plot(ex, ey, "o-", color="C0", lw=2, label="eval win")
        if evals[0]["step"] == 0:
            ax.axhline(evals[0]["win"], color="C3", ls="--", lw=1,
                       alpha=0.7, label=f"baseline ({evals[0]['win']:.3f})")
    ax.axhline(0.5, color="black", lw=0.5, alpha=0.3)
    ax.set_xlabel("step")
    ax.set_ylabel("win rate")
    ax.set_ylim(0, 1)
    ax.set_title("Win rate vs eval pool")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    # ----- (1,0) Eval margin -------------------------------------------------
    ax = axes[1, 0]
    if evals:
        ex = [e["step"] for e in evals]
        em = [e["margin"] for e in evals]
        ax.plot(ex, em, "o-", color="C2", lw=2)
        if evals[0]["step"] == 0:
            ax.axhline(evals[0]["margin"], color="C3", ls="--", lw=1,
                       alpha=0.7, label=f"baseline ({evals[0]['margin']:+d})")
            ax.legend(loc="best", fontsize=8)
    ax.axhline(0, color="black", lw=0.5, alpha=0.3)
    ax.set_xlabel("step")
    ax.set_ylabel("margin (ships)")
    ax.set_title("Eval score margin (summed across games)")
    ax.grid(alpha=0.3)

    # ----- (1,1) MLP diagnostics --------------------------------------------
    ax = axes[1, 1]
    if evals:
        ex = [e["step"] for e in evals]
        fr = [e["flip_rate"] * 100 for e in evals]
        dl = [e["mean_max_delta"] for e in evals]
        ax.plot(ex, fr, "o-", color="C4", lw=2, label="flip rate (%)")
        ax.set_ylabel("flip rate (%)", color="C4")
        ax.tick_params(axis="y", labelcolor="C4")
        ax.set_ylim(bottom=0)
        ax2 = ax.twinx()
        ax2.plot(ex, dl, "s-", color="C1", lw=2, label="|delta|max")
        ax2.set_ylabel("mean |delta|max", color="C1")
        ax2.tick_params(axis="y", labelcolor="C1")
        ax2.set_ylim(bottom=0)
    ax.set_xlabel("step")
    ax.set_title("MLP diagnostics  (is the network changing decisions?)")
    ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=110)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
