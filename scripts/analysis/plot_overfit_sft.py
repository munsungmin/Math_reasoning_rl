"""Plot observed SFT loss and greedy generation accuracy from overfit_four.py."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run", type=Path, nargs="+", help="SFT stages in training order"
    )
    args = parser.parse_args()
    train, evaluations, steps = [], [], []
    offset = 0
    for run in args.run:
        stage = [
            json.loads(line)
            for line in (run / "metrics.jsonl").read_text().splitlines()
        ]
        train.extend({**r, "step": r["step"] + offset} for r in stage)
        for path in sorted(run.glob("greedy_step_*.summary.json")):
            r = json.loads(path.read_text())
            evaluations.append(r)
            steps.append(int(r["label"].rsplit("_", 1)[1]) + offset)
        offset += max(r["step"] for r in stage)
    out = args.run[-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    axes[0].plot(
        [r["step"] for r in train], [r["loss"] for r in train], color="#2467A5", lw=2.4
    )
    axes[0].set(
        xlabel="Cumulative SFT optimizer update",
        ylabel="Response token cross entropy (log scale)",
        title="Loss decreases with gold-solution training",
        yscale="log",
    )
    axes[1].plot(
        steps, [r["accuracy"] for r in evaluations], "o-", color="#208563", lw=2.4
    )
    for step, r in zip(steps, evaluations, strict=True):
        axes[1].annotate(
            f"{r['correct']}/{r['samples']}",
            (step, r["accuracy"]),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
        )
    axes[1].set(
        xlabel="Cumulative SFT optimizer update",
        ylabel="Accuracy on original prompts",
        title="Generated answers still need separate evaluation",
        ylim=(-0.07, 1.15),
    )
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_xticks(steps)
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Four-problem SFT recovery: measured loss and greedy answer accuracy",
        fontsize=14,
    )
    for ext in ("png", "pdf"):
        fig.savefig(out / f"learning_curve.{ext}", dpi=180)
    plt.close(fig)
    with (out / "greedy_accuracy.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["sft_step", "correct", "samples", "accuracy", "length_limit_rate"]
        )
        for step, r in zip(steps, evaluations, strict=True):
            writer.writerow(
                [
                    step,
                    r["correct"],
                    r["samples"],
                    r["accuracy"],
                    r["length_limit_rate"],
                ]
            )


if __name__ == "__main__":
    main()
