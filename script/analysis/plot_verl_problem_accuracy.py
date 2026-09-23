"""Plot measured accuracy over time for each problem in a small verl run."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import pandas as pd
from omegaconf import OmegaConf


LABELS = {
    "test/prealgebra/1353.json": "54 cards: valid player counts",
    "test/number_theory/1055.json": "Gold coins: redistribute 7 bags into 8",
    "test/algebra/1035.json": "Treeks and squigs: weight equations",
    "test/precalculus/1202.json": "Arccos sum: cubic coefficients",
}


def analyze(run, window=20, through_step=None):
    run = Path(run).resolve()
    cfg = OmegaConf.load(run / "config.yaml")
    metadata = {
        record["prompt"][0]["content"]: {
            "sample_id": record["extra_info"]["sample_id"],
            "level": int(record["extra_info"]["level"]),
            "answer": record["reward_model"]["ground_truth"],
        }
        for record in pd.read_parquet(run / "train.parquet").to_dict("records")
    }
    files = {}
    for path in sorted(run.glob("sessions/*/rollouts/*.jsonl")):
        step = int(path.stem)
        if through_step is None or step <= through_step:
            files[step] = path
    records = []
    series = defaultdict(list)
    expected = cfg.actor_rollout_ref.rollout.n
    for step, path in sorted(files.items()):
        lines = path.read_text().splitlines(keepends=True)
        samples = [json.loads(line) for line in lines if line.endswith("\n")]
        if len(samples) != cfg.data.train_batch_size * expected:
            continue
        groups = defaultdict(list)
        for sample in samples:
            groups[sample["input"]].append(sample)
        for prompt, group in groups.items():
            if len(group) != expected:
                raise ValueError(f"Unexpected group size at step {step}")
            info = metadata[prompt]
            if any(sample["acc"] not in (0, 1) for sample in group):
                raise ValueError(f"Nonbinary accuracy at step {step}")
            record = {
                **info,
                "step": step,
                "samples": len(group),
                "correct": int(sum(sample["acc"] for sample in group)),
            }
            record["accuracy"] = record["correct"] / record["samples"]
            records.append(record)
            series[info["sample_id"]].append(record)
    if not records:
        raise ValueError("No complete rollout batches found")
    if len(series) > 8:
        raise ValueError("This figure is intended for small overfitting runs")
    last_step = max(record["step"] for record in records)
    out = run / "report" / f"problem_accuracy_step_{last_step}"
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "through_step": last_step,
        "window_steps": window,
        "samples_per_problem_per_step": expected,
        "source": "Training rollouts, sampled before each optimizer update",
        "problems": [],
    }

    def aggregate(items):
        total = sum(item["samples"] for item in items)
        correct = sum(item["correct"] for item in items)
        return {
            "from_step": items[0]["step"],
            "to_step": items[-1]["step"],
            "correct": correct,
            "samples": total,
            "accuracy": correct / total,
        }

    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    nrows = (len(series) + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(13, 4.1 * nrows), squeeze=False)
    colors = ["#0072B2", "#D55E00", "#009E73", "#8A4C97"]
    for index, (sample_id, items) in enumerate(sorted(series.items())):
        ax = axes.flat[index]
        steps = [item["step"] for item in items]
        smooth = []
        for item in items:
            tail = [r for r in items if item["step"] - window < r["step"] <= item["step"]]
            value = aggregate(tail)["accuracy"] if len(tail) == window else None
            item["rolling_accuracy"] = value
            smooth.append(value)
        first, last = aggregate(items[:window]), aggregate(items[-window:])
        windows = [
            aggregate(items[start:start + window])
            for start in range(0, len(items), window)
        ]
        info = {
            "sample_id": sample_id,
            "label": LABELS.get(sample_id, sample_id),
            "answer": items[0]["answer"],
            "first_window": first,
            "last_window": last,
            "change_percentage_points": 100 * (last["accuracy"] - first["accuracy"]),
            "nonoverlapping_windows": windows,
        }
        summary["problems"].append(info)
        ax.plot(steps, [item["accuracy"] for item in items], color="#7D8793", alpha=0.28, lw=0.8,
                label=f"Each step ({expected} responses)")
        ax.plot(steps, smooth, color=colors[index % len(colors)], lw=2.5,
                label=f"Trailing {window} steps ({window * expected} responses)")
        ax.scatter([last_step], [last["accuracy"]], color=colors[index % len(colors)], s=26, zorder=4)
        ax.set_title(f"{info['label']} | answer = {info['answer']}\n"
                     f"First {window}: {first['accuracy']:.1%}  |  Latest {window}: {last['accuracy']:.1%}",
                     loc="left", fontsize=11)
        ax.set(xlabel="Training step", ylabel="Answer accuracy", ylim=(-0.025, 1.025),
               xlim=(1, last_step + max(2, last_step * 0.02)))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.grid(axis="y", alpha=0.15)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axes.flat)[len(series):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle(f"Four-problem overfitting: training accuracy through step {last_step}",
                 fontsize=16, y=0.99)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95), h_pad=2.0)
    for extension in ("png", "pdf"):
        fig.savefig(out / f"problem_accuracy.{extension}", dpi=180)
    plt.close(fig)
    with (out / "accuracy_by_step.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda row: (row["step"], row["sample_id"])))
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(out), **summary}, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--through-step", type=int)
    args = parser.parse_args()
    if args.window < 1:
        parser.error("--window must be positive")
    analyze(args.run, args.window, args.through_step)
