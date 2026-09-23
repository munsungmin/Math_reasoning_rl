"""Independent generation/classification evaluation and saved-prediction re-scoring."""

from pathlib import Path

from hydra.utils import instantiate

from scripts.utils.io import read_rows, write_json, write_rows
from scripts.utils.metric import build_metrics


def evaluate_records(rows, metrics_config):
    metrics = build_metrics(metrics_config)
    metrics.update(rows)
    return metrics.compute()


def grade_records(rows, grader, batch_size=8):
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        judgements = grader(
            [
                {
                    "dataset": r.get("dataset", "math"),
                    "completion": r["output"],
                    "answer": r["answer"],
                }
                for r in batch
            ]
        )
        for row, judgement in zip(batch, judgements, strict=True):
            row.update(judgement)
    return rows


def run(config):
    output = Path(config.runtime.output_dir)
    if config.evaluation.predictions:
        rows = read_rows(config.evaluation.predictions)
        if config.stage == "predict":
            write_rows(output / "generated.jsonl", rows)
            write_json(
                output / "result.json", {"stage": "predict", "records": len(rows)}
            )
            return
        if config.evaluation.kind == "solve":
            grade_records(rows, instantiate(config.grader))
        summary = evaluate_records(rows, config.task.test_metrics)
    else:
        from scripts.reasoning.model.loading import load_model

        model, tokenizer, spec = load_model(
            config.model,
            trainable=False,
            device="cuda" if config.runtime.gpu_ids else "cpu",
        )
        # Inference freezes parameters; WeightNorm's selection still describes trainable
        # adaptation parameters, so for observation expose the selected adapter explicitly.
        from scripts.reasoning.model.observation import observe_loaded_model

        observe_loaded_model(
            model, config.callbacks, output / "observations", stage=config.stage
        )
        data = instantiate(config.data)
        examples = data.rows(config.evaluation.split)
        if not examples:
            raise ValueError("Evaluation requires a nonempty configured split")
        if config.evaluation.kind == "classify":
            from .inference.classification import classify

            rows = list(
                classify(
                    model,
                    tokenizer,
                    examples,
                    labels=list(config.evaluation.labels),
                    batch_size=config.test_config.batch_size,
                )
            )
        else:
            from scripts.reasoning.rollout.generation import generate

            rows = list(
                generate(model, tokenizer, examples, instantiate(config.test_config))
            )
        write_rows(output / "generated.jsonl", rows)
        if config.stage == "predict":
            write_json(
                output / "result.json",
                {"stage": "predict", "records": len(rows), "model": vars(spec)},
            )
            return
        if config.evaluation.kind == "solve":
            grade_records(rows, instantiate(config.grader))
        summary = evaluate_records(rows, config.task.test_metrics)
        summary["model"] = vars(spec)
        summary["split"] = config.evaluation.split
        # Keep benchmark overlap visible; no checkpoint selection is done here.
        train_ids = {r["sample_id"] for r in data.rows("train")}
        summary["training_overlap"] = len({r["sample_id"] for r in rows} & train_ids)
    write_rows(output / "predictions.jsonl", rows)
    write_json(output / "summary.json", summary)
    write_json(
        output / "result.json",
        {"completed": True, "summary": str(output / "summary.json")},
    )
    print(summary, flush=True)
