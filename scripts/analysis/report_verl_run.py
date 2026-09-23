"""Export measured native verl results, including incomplete runs."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from omegaconf import OmegaConf


def rows(path):
    with path.open() as stream:
        for line in stream:
            # A live writer may not have finished the last line yet.
            if line.endswith("\n") and line.strip():
                yield json.loads(line)


def mean(values):
    return sum(values) / len(values) if values else None


def summarize(run, output=None):
    run = Path(run).resolve()
    output = Path(output) if output else run / "report"
    output.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(run / "config.yaml")
    target = cfg.trainer.total_training_steps
    batch = cfg.data.train_batch_size * cfg.actor_rollout_ref.rollout.n
    metrics = {}
    metric_path = run / "metrics.jsonl"
    if metric_path.exists():
        for event in rows(metric_path):
            if event.get("phase") != "test":
                metrics[event["step"]] = event["data"]

    # Native per-step files are authoritative; the launcher's cumulative log
    # can miss a file that its asynchronous writer finishes after the metric.
    train_files = {}
    validation_files = {}
    test_files = []
    for session in sorted((run / "sessions").glob("*")):
        if not session.is_dir():
            continue
        for path in (session / "rollouts").glob("*.jsonl"):
            train_files[int(path.stem)] = path
        for path in (session / "validation").glob("*.jsonl"):
            validation_files[int(path.stem)] = path
        test_files.extend(sorted((session / "test").glob("*.jsonl")))

    records = []
    problems = defaultdict(list)
    for step, path in sorted(train_files.items()):
        samples = list(rows(path))
        if len(samples) != batch:
            continue
        event = metrics.get(step, {})
        records.append(
            {
                "step": step,
                "rollouts": len(samples),
                "accuracy": mean([row["acc"] for row in samples]),
                "reward": mean([row["score"] for row in samples]),
                "format_rate": mean([row["format"] for row in samples]),
                "numeric_rate": mean([row["numeric"] for row in samples]),
                "loss": event.get("actor/loss"),
                "kl_loss": event.get("actor/kl_loss"),
                "grad_norm": event.get("actor/grad_norm"),
                "response_length": event.get("response_length/mean"),
                "length_cap_fraction": event.get("response_length/clip_ratio"),
                "step_seconds": event.get("timing_s/step"),
            }
        )
        grouped = defaultdict(list)
        for row in samples:
            grouped[row["input"]].append(row)
        for prompt, group in grouped.items():
            problems[prompt].append(
                {
                    "step": step,
                    "samples": len(group),
                    "correct": sum(row["acc"] for row in group),
                    "reward_sum": sum(row["score"] for row in group),
                    "answer": group[0]["gts"],
                }
            )

    def evaluation(path):
        samples = list(rows(path))
        return {
            "step": int(path.stem),
            "samples": len(samples),
            "correct": sum(row["acc"] for row in samples),
            "accuracy": mean([row["acc"] for row in samples]),
            "reward": mean([row["score"] for row in samples]),
        }

    validation = [evaluation(path) for _, path in sorted(validation_files.items())]
    test = evaluation(test_files[-1]) if test_files else None
    last_step = records[-1]["step"] if records else 0
    last_checkpoint = Path(cfg.trainer.default_local_dir) / "last"
    checkpoint_step = (
        int(last_checkpoint.resolve().name.removeprefix("global_step_"))
        if last_checkpoint.is_symlink()
        else None
    )
    window = min(100, len(records))
    windows = {}
    for name, subset in (("first", records[:window]), ("last", records[-window:])):
        windows[name] = {
            "steps": len(subset),
            **{
                key: mean([row[key] for row in subset if row[key] is not None])
                for key in ("accuracy", "reward", "format_rate", "length_cap_fraction")
            },
        }
    seconds = [r["step_seconds"] for r in records[-100:] if r["step_seconds"]]
    summary = {
        "run": str(run),
        "training_complete": (
            len(records) == target and last_step == target and checkpoint_step == target
        ),
        "target_steps": target,
        "completed_steps": len(records),
        "latest_step": last_step,
        "target_rollouts": target * batch,
        "recorded_rollouts": sum(row["rollouts"] for row in records),
        "distinct_train_problems": len(problems),
        "checkpoint_step": checkpoint_step,
        "first_window": windows["first"],
        "last_window": windows["last"],
        "last_step_metrics": records[-1] if records else None,
        "validation": validation,
        "test": test,
        "estimated_training_hours_remaining": (
            (target - last_step) * mean(seconds) / 3600 if seconds else None
        ),
        "nonfinite_metrics": [
            {"step": r["step"], "metric": key}
            for r in records
            for key, value in r.items()
            if isinstance(value, float) and not math.isfinite(value)
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    if records:
        with (output / "learning_curve.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    with (output / "train_problems.jsonl").open("w") as stream:
        for index, (prompt, entries) in enumerate(problems.items(), 1):
            total = sum(r["samples"] for r in entries)
            tail = entries[-window:]
            stream.write(
                json.dumps(
                    {
                        "problem": index,
                        "prompt": prompt,
                        "answer": entries[0]["answer"],
                        "rollouts": total,
                        "accuracy": sum(r["correct"] for r in entries) / total,
                        "last_window_accuracy": sum(r["correct"] for r in tail)
                        / sum(r["samples"] for r in tail),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    def percent(value):
        return f"{value:.2%}" if value is not None else "pending"

    lines = [
        f"# {cfg.launcher.experiment}",
        "",
        f"Run: `{run}`",
        "",
        f"Training: {'complete' if summary['training_complete'] else 'incomplete'}; "
        f"{len(records):,}/{target:,} steps, "
        f"{summary['recorded_rollouts']:,}/{target * batch:,} training responses.",
        f"Latest complete checkpoint: {checkpoint_step}.",
        "",
        f"Train accuracy, first/last {window} steps: "
        f"{percent(windows['first']['accuracy'])} / {percent(windows['last']['accuracy'])}.",
        f"Latest validation accuracy: {percent(validation[-1]['accuracy']) if validation else 'pending'}.",
        f"Final test accuracy: {percent(test['accuracy']) if test else 'pending'}.",
        "",
        "Train accuracy is measured on sampled training responses before each update. "
        "Validation and test use greedy decoding on separate held-out subsets. "
        "These are internal subset results, not the official full MATH-500 score.",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n")
    if records:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
        for ax, key, label in zip(
            axes.flat,
            ["accuracy", "reward", "loss", "length_cap_fraction"],
            [
                "Answer accuracy",
                "Mean reward",
                "Actor loss",
                "Response length cap rate",
            ],
        ):
            pairs = [(r["step"], r[key]) for r in records if r[key] is not None]
            x, y = zip(*pairs) if pairs else ([], [])
            ax.plot(x, y, alpha=0.25, linewidth=0.6, color="#0072B2")
            smooth = [mean(y[max(0, i - 99) : i + 1]) for i in range(len(y))]
            ax.plot(x, smooth, color="#0072B2", label="Train, 100-step moving mean")
            if key == "accuracy":
                ax.plot(
                    [v["step"] for v in validation],
                    [v["accuracy"] for v in validation],
                    color="#D55E00",
                    marker=".",
                    label="Held-out validation",
                )
                ax.legend(fontsize=8)
            if key in ("accuracy", "length_cap_fraction"):
                ax.set_ylim(-0.02, 1.02)
            ax.set(xlabel="Training step", ylabel=label)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"{cfg.launcher.experiment} | {last_step:,}/{target:,} steps")
        for extension in ("png", "pdf"):
            fig.savefig(output / f"learning_curve.{extension}", dpi=180)
        plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.run, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
