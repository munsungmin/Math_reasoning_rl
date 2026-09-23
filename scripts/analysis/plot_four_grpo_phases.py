"""Plot measured pure-GRPO accuracy across the two continuation phases."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    args = parser.parse_args()
    root = args.artifact_root
    phases = [
        json.loads((root / name).read_text())
        for name in ("progress_lr3e4.json", "progress_entropy_m001.json")
    ]
    labels = {
        "test/prealgebra/1353.json": "Cards",
        "test/number_theory/1055.json": "Coins",
        "test/algebra/1035.json": "Weights",
        "test/precalculus/1202.json": "Arccos",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True)
    records = []
    boundary = len(phases[0]["per_step"])
    for ax, (problem, title) in zip(axes.flat, labels.items(), strict=True):
        offset = 0
        for phase_name, report, color in zip(
            ("A", "B"), phases, ("#2563eb", "#ea580c"), strict=True
        ):
            steps = report["per_step"]
            xs = [offset + s["step"] for s in steps]
            ys = [s["per_problem"][problem]["correct"] / 8 for s in steps]
            rolling = [
                sum(ys[max(0, i - 9) : i + 1]) / len(ys[max(0, i - 9) : i + 1])
                for i in range(len(ys))
            ]
            ax.plot(
                xs, ys, color=color, alpha=0.2, linewidth=0.8, marker=".", markersize=3
            )
            ax.plot(
                xs,
                rolling,
                color=color,
                linewidth=2,
                label=f"Phase {phase_name}: trailing 10 updates",
            )
            records.extend(
                dict(
                    phase=phase_name,
                    phase_step=s["step"],
                    continuation_step=x,
                    sample_id=problem,
                    correct=s["per_problem"][problem]["correct"],
                    samples=8,
                    accuracy=y,
                    trailing_window_accuracy=avg,
                )
                for s, x, y, avg in zip(steps, xs, ys, rolling, strict=True)
            )
            offset += len(steps)
        ax.axvline(boundary + 0.5, color="#6b7280", linestyle="--", linewidth=1)
        ax.axhline(1, color="#9ca3af", linewidth=0.7)
        ax.set_title(title)
        ax.set_ylim(0, 1.04)
        ax.grid(axis="y", alpha=0.18)
    for ax in axes[:, 0]:
        ax.set_ylabel("Training accuracy")
    for ax in axes[-1]:
        ax.set_xlabel("Additional RL updates after the original 900 + pilot 4")
    axes[0, 0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Pure GRPO on the original four problems", fontsize=16)
    fig.text(
        0.5,
        0.92,
        "G=8, temperature=1, response cap=1024; entropy penalty added after A-50",
        ha="center",
        fontsize=10,
    )
    fig.text(
        0.5,
        0.015,
        "Faint dots: 8 responses per problem per update. Lines: up to 80 responses; window restarts at phase B.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.91))
    out = root / "report"
    out.mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"pure_grpo_phases.{suffix}", dpi=180)
    with (out / "pure_grpo_phases.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(out / "pure_grpo_phases.png")


if __name__ == "__main__":
    main()
