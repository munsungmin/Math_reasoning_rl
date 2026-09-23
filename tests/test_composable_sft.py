"""CPU integration tests: actual Lightning/PEFT training, restart, and inference."""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning.pytorch")
pytest.importorskip("hydra")

from scripts.utils.config import compose_experiment
from scripts.utils.io import write_rows


@pytest.fixture
def tiny_artifacts(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    torch.set_num_threads(1)
    torch.manual_seed(3)
    vocabulary = {
        word: i
        for i, word in enumerate(
            [
                "<pad>",
                "<eos>",
                "<unk>",
                "A",
                "B",
                "C",
                "D",
                "question",
                "answer",
                "train",
                "validation",
                "test",
                "0",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
            ]
        )
    }
    raw = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>"
    )
    path = tmp_path / "model"
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=len(vocabulary),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=128,
            eos_token_id=1,
            pad_token_id=0,
            tie_word_embeddings=True,
        )
    )
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    for split, count in (("train", 8), ("validation", 3), ("test", 3)):
        write_rows(
            tmp_path / f"{split}.jsonl",
            [
                {
                    "sample_id": f"{split}-{i}",
                    "family_id": f"{split}-{i}",
                    "prompt": f"question {split} {i}",
                    "target": f"answer {i}",
                    "answer": str(i),
                    "label": i % 4,
                }
                for i in range(count)
            ],
        )
    return tmp_path


def recipe(artifacts, experiment, output):
    return compose_experiment(
        [
            f"experiment={experiment}",
            f"model.initialization.path={artifacts / 'model'}",
            "model.dtype=float32",
            "model.gradient_checkpointing=false",
            "model.adapter.rank=2",
            "model.adapter.alpha=4",
            "runtime.gpu_ids=[]",
            "runtime.num_workers=0",
            f"runtime.output_dir={output}",
            f"data.sources.train={artifacts / 'train.jsonl'}",
            f"data.sources.validation={artifacts / 'validation.jsonl'}",
            f"data.sources.test={artifacts / 'test.jsonl'}",
            "data.expected_counts=null",
            "data.levels=null",
            "callbacks=analysis",
            "trainer.save_every=1",
            "test_config.max_new_tokens=3",
            "test_config.batch_size=2",
        ]
        + (
            [
                "trainer.accelerator=cpu",
                "trainer.precision=32-true",
                "train_config.batch_size=2",
                "train_config.accumulate_grad_batches=1",
                "train_config.max_steps=4",
                "train_config.max_epochs=2",
                "train_config.max_tokens=32",
            ]
            if experiment.startswith("sft_")
            else []
        )
    )


@pytest.mark.parametrize("kind", ["solve", "classify"])
def test_sft_export_and_independent_evaluation(tiny_artifacts, kind):
    from scripts.eval.pipeline import run as evaluate
    from scripts.train.sft.runner import run

    output = tiny_artifacts / f"sft-{kind}"
    output.mkdir()
    cfg = recipe(tiny_artifacts, f"sft_{kind}", output)
    run(cfg)
    result = json.loads((output / "result.json").read_text())
    assert result["global_step"] == 4
    assert Path(result["export"], "adapter_model.safetensors").is_file()
    assert (output / "observations/metrics.jsonl").is_file()
    eval_output = tiny_artifacts / "eval"
    evaluation = recipe(
        tiny_artifacts, f"eval_{'math' if kind == 'solve' else 'classify'}", eval_output
    )
    evaluation.model.initialization.path = result["export"]
    evaluate(evaluation)
    summary = json.loads((eval_output / "summary.json").read_text())
    assert summary["samples"] == 3
    assert summary["training_overlap"] == 0


def test_sft_interrupted_resume_matches_uninterrupted(tiny_artifacts, monkeypatch):
    import lightning.pytorch as pl
    from safetensors.torch import load_file

    from scripts.train.sft.runner import run

    full, resumed = tiny_artifacts / "full", tiny_artifacts / "resumed"
    full.mkdir()
    resumed.mkdir()
    cfg = recipe(tiny_artifacts, "sft_solve", full)
    run(cfg)
    original = pl.Trainer

    class StopAtTwo(pl.Callback):
        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            if trainer.global_step == 2:
                raise RuntimeError("simulate worker interruption")

    def interrupted_trainer(**kwargs):
        kwargs["callbacks"].append(StopAtTwo())
        return original(**kwargs)

    monkeypatch.setattr(pl, "Trainer", interrupted_trainer)
    cfg = recipe(tiny_artifacts, "sft_solve", resumed)
    with pytest.raises(RuntimeError, match="simulate worker interruption"):
        run(cfg)
    assert (
        torch.load(resumed / "checkpoints/last.ckpt", weights_only=False)["global_step"]
        == 2
    )
    monkeypatch.setattr(pl, "Trainer", original)
    cfg.resume = str(resumed / "checkpoints/last.ckpt")
    run(cfg)
    a, b = [json.loads((p / "result.json").read_text()) for p in (full, resumed)]
    assert a["consumed_ids"] == b["consumed_ids"]
    weights_a, weights_b = [
        load_file(Path(r["export"]) / "adapter_model.safetensors") for r in (a, b)
    ]
    for key in weights_a:
        torch.testing.assert_close(weights_a[key], weights_b[key], rtol=0, atol=0)
