"""Export measured reward and per-arm learning curves; never invent missing metrics."""

import collections
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

BASE = Path("/data/sungmin/math_reasoning")

OUT = BASE / "reports/math_error_transfer/current"


def save(fig, name):
    for fmt in ["png", "svg", "pdf"]:
        fig.savefig(OUT / f"{name}.{fmt}", bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )

    p = BASE / "artifacts/reward_gate/20260910-025950"

    rows = [
        json.loads(line) for line in (p / "graded_v2.jsonl").read_text().splitlines()
    ]

    g = collections.defaultdict(list)

    for r in rows:
        g[r["id"]].append(r)

    fig, axs = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")

    axs[0].bar(
        range(1, 17),
        [sum(r["correct"] for r in rs) for rs in g.values()],
        color="#0072B2",
    )

    axs[0].set(
        xlabel="Problem index (fixed gate sample)",
        ylabel="Correct responses / 8",
        ylim=(0, 8.7),
        xticks=[1, 4, 8, 12, 16],
        title="Initial answer reward signal",
    )

    c = collections.Counter(r["reward"] for r in rows)

    xs = sorted(c)

    axs[1].bar(xs, [c[x] for x in xs], width=0.6, color="#009E73")

    axs[1].set(
        xlabel="Total reward",
        ylabel="Trajectories",
        xticks=[0, 1, 2, 3],
        title="Reward distribution",
    )

    axs[2].hist(
        [r["completion_tokens"] for r in rows],
        bins=range(0, 769, 64),
        color="#E69F00",
    )

    axs[2].axvline(768, color="#333333", ls="--", lw=1)

    axs[2].set(
        xlabel="Completion tokens",
        ylabel="Trajectories",
        title="Length cap: 54/128 (42.2%)",
    )

    fig.suptitle(
        "Qwen2.5-Math-1.5B | MATH-500 levels 3–5 | 16 problems × 8 samples",
        fontsize=12,
    )

    save(fig, "initial_reward_gate")

    all_records = []

    for f in sorted(
        (BASE / "artifacts/math_error_transfer/main").glob(
            "*/*/*/math500_level3_5/rollouts.jsonl"
        )
    ):
        arm = f.parents[3].name
        seed = f.parents[2].name
        steps = collections.defaultdict(list)
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            steps[r["global_step"] + 1].append(r)
        records = []
        for step, rs in sorted(steps.items()):
            if len(rs) != 32:
                continue
            row = dict(
                arm=arm,
                seed=seed,
                run=f.parents[1].name,
                update=step,
                accuracy=sum(r["correctness"] for r in rs) / 32,
                mean_reward=sum(r["reward"] for r in rs) / 32,
                mean_completion_tokens=sum(r["completion_tokens"] for r in rs) / 32,
                length_cap_fraction=sum(r["completion_tokens"] >= 768 for r in rs) / 32,
            )
            records.append(row)
            all_records.append(row)
        if not records:
            continue
        fig, axs = plt.subplots(1, 3, figsize=(12, 3.4), layout="constrained")
        x = [r["update"] for r in records]
        for ax, key, label in zip(
            axs,
            ["accuracy", "mean_reward", "length_cap_fraction"],
            ["Answer correctness", "Mean total reward", "Length cap fraction"],
        ):
            y = [r[key] for r in records]
            ax.plot(x, y, color="#0072B2", alpha=0.3, lw=0.8)
            smooth = [
                sum(y[max(0, i - 9) : i + 1]) / len(y[max(0, i - 9) : i + 1])
                for i in range(len(y))
            ]
            ax.plot(x, smooth, color="#0072B2", lw=1.8, label="10-update rolling mean")
            ax.set(xlabel="Optimizer update", ylabel=label, xlim=(1, 300))
            ax.legend(fontsize=8)
            if key != "mean_reward":
                ax.set_ylim(-0.02, 1.02)
        fig.suptitle(
            f"{arm} | {seed} | Qwen2.5-Math-1.5B | train data only", fontsize=12
        )
        save(fig, f"{arm}_{seed}_learning")

    if all_records:
        with (OUT / "learning_curves.csv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=list(all_records[0]))
            w.writeheader()
            w.writerows(all_records)

    (OUT / "README.md").write_text(
        "Measured training curves and initial reward diagnostic only. Held-out pass@K and trajectory validity require their separate generation/judgement files. Gray or missing results are never fabricated.\n"
    )

    print(OUT)

    by_arm = collections.defaultdict(list)

    for f in sorted(
        (BASE / "artifacts/math_error_transfer/evaluation").glob("*/*/summary.json")
    ):
        r = json.loads(f.read_text())
        by_arm[r["arm"]].append(r)

    summary_rows = []

    for arm, rs in by_arm.items():
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
        for j, r in enumerate(sorted(rs, key=lambda r: r["seed"])):
            color = ["#0072B2", "#D55E00"][j % 2]
            label = f"seed {r['seed']}"
            v = r["greedy_accuracy"]
            lo, hi = v["ci95"]
            axs[0].bar(j, v["mean"], color=color, width=0.55)
            axs[0].errorbar(
                j,
                v["mean"],
                yerr=[[v["mean"] - lo], [hi - v["mean"]]],
                color="#333333",
                capsize=4,
            )
            ks = [int(k) for k in r["pass_at_k"]]
            ys = [r["pass_at_k"][str(k)]["mean"] for k in ks]
            lows = [r["pass_at_k"][str(k)]["ci95"][0] for k in ks]
            highs = [r["pass_at_k"][str(k)]["ci95"][1] for k in ks]
            axs[1].plot(ks, ys, "o-", color=color, label=label)
            axs[1].fill_between(ks, lows, highs, color=color, alpha=0.12)
            for k, y in zip(ks, ys):
                summary_rows.append(
                    dict(
                        arm=arm,
                        seed=r["seed"],
                        k=k,
                        pass_at_k=y,
                        greedy_accuracy=v["mean"],
                    )
                )
        axs[0].set(
            xticks=range(len(rs)),
            xticklabels=[
                f"seed {r['seed']}" for r in sorted(rs, key=lambda r: r["seed"])
            ],
            ylim=(0, 1),
            ylabel="Held-out answer accuracy",
            title="Greedy: 67 problems",
        )
        axs[1].set(
            xticks=[1, 2, 4],
            ylim=(0, 1),
            xlabel="K",
            ylabel="pass@K",
            title="Sampled: 30 problems × 4",
        )
        axs[1].legend()
        fig.suptitle(
            f"{arm} | Answer metrics (validity judgement pending)", fontsize=12
        )
        save(fig, f"{arm}_heldout_pass_k")

    if summary_rows:
        with (OUT / "heldout_answer_metrics.csv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
            w.writeheader()
            w.writerows(summary_rows)


if __name__ == "__main__":
    main()
