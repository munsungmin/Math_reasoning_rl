"""Contracts that matter when changing composition, inputs and execution backends."""

import random
import subprocess
import sys

import pytest

pytest.importorskip("hydra")
from omegaconf import OmegaConf

from scripts.utils.config import compose_experiment
from scripts.utils.paths import REPO


@pytest.mark.parametrize(
    "experiment", ["rl_math", "sft_solve", "sft_classify", "eval_math", "eval_classify"]
)
def test_configuration_composes_without_loading_training_frameworks(experiment):
    code = f"""import sys
from scripts.utils.config import compose_experiment, validate
c=compose_experiment(['experiment={experiment}']); validate(c)
assert 'torch' not in sys.modules
assert 'lightning' not in sys.modules
assert not {{'model', 'optimizer', 'loss', 'trainer', 'callbacks', 'data'}} & set(c.task)
"""
    subprocess.run([sys.executable, "-c", code], cwd=REPO, check=True)


def test_native_configuration_preserves_existing_algorithm(tmp_path):
    from scripts.train.rl.config import to_native
    from scripts.train.rl.native_config import build_config, load_settings

    project = compose_experiment()
    previous = build_config(
        tmp_path / "run-parity", settings=load_settings(REPO / "configs/verl_math.yaml")
    )
    current = build_config(
        tmp_path / "run-parity", settings=to_native(project, resolve_model=False)
    )
    old, new = [OmegaConf.to_container(c, resolve=True) for c in (previous, current)]
    # Only run naming/provenance and reward record plumbing intentionally change.
    for value in (old, new):
        for key in ("runtime", "launcher", "math_reasoning"):
            value.pop(key, None)
        value["trainer"].pop("default_local_dir")
        value["trainer"].pop("logger")
        value["reward"].pop("custom_reward_function")
        value["reward"].pop("reward_manager")
    assert old == new


def test_metric_contracts_and_distributed_padding():
    from scripts.eval.pipeline import evaluate_records
    from scripts.utils.metric import build_metrics

    config = compose_experiment(["experiment=eval_math"])
    records = [
        {
            "sample_id": identity,
            "sample_index": i,
            "correct": correct,
            "generated_tokens": 4,
            "hit_length_limit": False,
        }
        for identity in ("a", "b")
        for i, correct in enumerate((True, False))
    ]
    values = evaluate_records(records + [records[0]], config.task.test_metrics)
    assert values["accuracy"] == 0.5
    assert values["samples"] == 4
    assert values["pass@1"] == 0.5
    with pytest.raises(ValueError, match="Conflicting"):
        evaluate_records(
            records + [dict(records[0], correct=False)], config.task.test_metrics
        )
    with pytest.raises(ValueError, match="missing|requires"):
        build_metrics(config.task.test_metrics).update([{"sample_id": "x"}])


def test_sampler_prefetch_does_not_advance_consumed_cursor():
    from scripts.reasoning.data.sampler import ResumableBatchSampler

    sampler = ResumableBatchSampler(11, 2, 42, world_size=2)
    order = list(sampler)
    sampler.acknowledge(len(order[0]))
    state = sampler.state_dict()
    resumed = ResumableBatchSampler(11, 2, 42, world_size=2)
    resumed.load_state_dict(state)
    assert list(resumed) == order[1:]
    assert state["cursor"] == 2
    with pytest.raises(ValueError, match="world_size"):
        ResumableBatchSampler(11, 2, 42, world_size=1).load_state_dict(state)


def test_loss_ignores_prompt_and_padding_gradients():
    torch = pytest.importorskip("torch")
    from scripts.reasoning.loss.supervised import CausalTokenLoss

    logits = torch.zeros((1, 6, 5), requires_grad=True)
    labels = torch.tensor([[-100, -100, -100, 2, 1, -100]])
    terms = CausalTokenLoss()(logits, labels)
    assert terms["target_tokens"] == 2
    assert float(terms["loss"].detach()) == pytest.approx(__import__("math").log(5))
    terms["loss"].backward()
    assert logits.grad[:, :2].count_nonzero() == 0
    assert logits.grad[:, 2:4].count_nonzero() > 0
    assert logits.grad[:, 4:].count_nonzero() == 0


def test_observation_preserves_model_and_rng(tmp_path):
    torch = pytest.importorskip("torch")
    import numpy as np
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from scripts.reasoning.model.observation import observe_loaded_model
    from scripts.reasoning.modules.adapters import attach_lora
    from scripts.utils.callbacks import build_observations
    from scripts.utils.logger import ResultWriter

    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
        )
    )
    model = attach_lora(
        model, {"rank": 2, "alpha": 4, "dropout": 0.1, "target_modules": ["q_proj"]}
    )
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.ones_like(p)
    before = {
        n: (
            p.detach().clone(),
            p.requires_grad,
            p.grad.clone() if p.grad is not None else None,
        )
        for n, p in model.named_parameters()
    }
    rng = torch.get_rng_state().clone()
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cfg = compose_experiment(["callbacks=analysis"])
    build_observations(cfg.callbacks).run(
        model, step=20, stage="train", writer=ResultWriter(tmp_path)
    )
    assert model.training
    assert torch.equal(rng, torch.get_rng_state())
    assert random.getstate() == python_state
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
    for name, p in model.named_parameters():
        value, requires_grad, gradient = before[name]
        assert torch.equal(value, p) and p.requires_grad == requires_grad
        if gradient is not None:
            assert torch.equal(gradient, p.grad)
    model.requires_grad_(False)
    observe_loaded_model(model, cfg.callbacks, tmp_path / "frozen", stage="test")
    assert not any(p.requires_grad for p in model.parameters())


def test_grpo_uses_masked_group_returns_and_native_policy_loss():
    torch = pytest.importorskip("torch")
    pytest.importorskip("verl")
    import numpy as np

    from scripts.reasoning.advantage.grpo import GRPOAdvantage
    from scripts.reasoning.loss.policy import PolicyObjective

    rewards = torch.tensor([[0.0, 0.0], [0.0, 2.0], [0.0, 1.0], [0.0, 1.0]])
    mask = torch.tensor([[1.0, 1.0]] * 4)
    advantages, _ = GRPOAdvantage()(rewards, mask, np.array(["a", "a", "b", "b"]))
    assert advantages[0, 0] == pytest.approx(-(2**-0.5), abs=1e-6)
    assert advantages[1, 0] == pytest.approx(2**-0.5, abs=1e-6)
    assert torch.count_nonzero(advantages[2:]) == 0
    log_probs = torch.zeros_like(mask, requires_grad=True)
    loss, _ = PolicyObjective().policy_loss(
        torch.zeros_like(mask), log_probs, advantages, mask
    )
    loss.backward()
    assert log_probs.grad[0, 0] > 0 and log_probs.grad[1, 0] < 0
    assert torch.count_nonzero(log_probs.grad[2:]) == 0


def test_replica_adapter_export_keeps_dtype_and_rejects_disagreement(tmp_path):
    torch = pytest.importorskip("torch")
    from scripts.train.rl.adapter import AdapterModelView, load_adapter_state
    from scripts.utils.io import write_json

    write_json(tmp_path / "fsdp_config.json", {"world_size": 2})
    write_json(tmp_path / "lora_train_meta.json", {"r": 2, "lora_alpha": 4})
    state = {
        "base_model.model.q_proj.lora_A.default.weight": torch.ones(
            (2, 4), dtype=torch.float32
        ),
        "base_model.model.q_proj.lora_B.default.weight": torch.ones(
            (4, 2), dtype=torch.float32
        ),
    }
    for rank in range(2):
        torch.save(state, tmp_path / f"model_world_size_2_rank_{rank}.pt")
    merged, metadata = load_adapter_state(tmp_path)
    assert all(torch.equal(merged[k], value) for k, value in state.items())
    assert list(AdapterModelView(merged, metadata).named_modules())[0][1].scaling == {
        "default": 2.0
    }
    state[next(iter(state))][0, 0] = 0.5
    torch.save(state, tmp_path / "model_world_size_2_rank_1.pt")
    with pytest.raises(ValueError, match="disagree"):
        load_adapter_state(tmp_path)


def test_sft_adapter_initialization_keeps_weights_and_uses_rl_dropout(tmp_path):
    import json

    from scripts.train.rl.adapter import prepare_initial_adapter
    from scripts.utils.io import write_json

    source, target = tmp_path / "sft", tmp_path / "rl"
    source.mkdir()
    write_json(
        source / "adapter_config.json",
        {"lora_dropout": 0.05, "r": 2, "lora_alpha": 4, "bias": "none"},
    )
    weights = source / "adapter_model.safetensors"
    weights.write_bytes(b"unchanged-adapter-weights")
    metadata = prepare_initial_adapter(source, target, "/base/model")
    assert metadata["source_dropout"] == 0.05
    assert (
        json.loads((source / "adapter_config.json").read_text())["lora_dropout"] == 0.05
    )
    assert json.loads((target / "adapter_config.json").read_text())["lora_dropout"] == 0
    assert (target / weights.name).read_bytes() == weights.read_bytes()


def test_conditional_probability_statistics_need_predictions():
    import math

    from scripts.reasoning.metric.probability import ConditionalEntropy, ConditionalKL

    record = {
        "sample_id": "x",
        "log_probs": [math.log(0.5), math.log(0.5)],
        "reference_log_probs": [math.log(0.5), math.log(0.5)],
    }
    entropy, kl = ConditionalEntropy(), ConditionalKL()
    entropy.update([record])
    kl.update([record])
    assert entropy.compute()["conditional_entropy"] == pytest.approx(math.log(2))
    assert kl.compute()["conditional_kl"] == pytest.approx(0.0)


def test_worker_cleanup_stops_its_descendants():
    import time
    from pathlib import Path

    from scripts.utils.process import stop_group

    program = "import signal,subprocess,sys; child=subprocess.Popen([sys.executable,'-c','import signal; signal.pause()']); print(child.pid,flush=True); signal.pause()"
    process = subprocess.Popen(
        [sys.executable, "-c", program],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        child = int(process.stdout.readline())
        stop_group(process, timeout=1)
        assert process.poll() is not None
        for _ in range(100):
            try:
                state = Path(f"/proc/{child}/stat").read_text().split()[2]
            except FileNotFoundError:
                break
            if state == "Z":
                break
            time.sleep(0.01)
        else:
            pytest.fail("Worker's descendant remained alive after cleanup")
    finally:
        stop_group(process, timeout=1)
        process.stdout.close()
