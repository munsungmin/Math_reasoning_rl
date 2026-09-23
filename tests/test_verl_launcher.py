"""Checkpoint rotation and CPU-only launcher contract checks."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from omegaconf import OmegaConf
from omegaconf.errors import ConfigAttributeError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "script"))
from train_verl import build_config, compute_score, load_settings, main, publish_last


class VerLCheckpointTest(unittest.TestCase):
    def make_checkpoint(self, root, step, world_size=4):
        checkpoint = root / f"global_step_{step}"
        actor = checkpoint / "actor"
        (actor / "huggingface").mkdir(parents=True)
        (checkpoint / "data.pt").write_bytes(b"dataloader")
        for rank in range(world_size):
            for kind in ("model", "optim", "extra_state"):
                (actor / f"{kind}_world_size_{world_size}_rank_{rank}.pt").write_bytes(
                    b"shard"
                )
        (actor / "fsdp_config.json").write_text(json.dumps({"world_size": world_size}))
        for name in (
            "lora_train_meta.json",
            "huggingface/config.json",
            "huggingface/tokenizer_config.json",
        ):
            (actor / name).write_text("{}")
        return checkpoint

    def test_complete_save_replaces_last_and_preserves_unrelated_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = self.make_checkpoint(root, 20)
            (root / "last").symlink_to(old.name)
            new = self.make_checkpoint(root, 40)
            future = root / "global_step_60"
            future.mkdir()
            unrelated = root / "export"
            unrelated.mkdir()
            (root / "latest_checkpointed_iteration.txt").write_text("40")
            self.assertEqual(publish_last(root), 40)
            self.assertEqual((root / "last").resolve(), new)
            self.assertFalse(old.exists())
            self.assertTrue(future.exists())
            self.assertTrue(unrelated.exists())
            self.assertIsNone(publish_last(root))

    def test_incomplete_save_keeps_previous_last(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = self.make_checkpoint(root, 20)
            (root / "last").symlink_to(old.name)
            new = self.make_checkpoint(root, 40)
            (new / "actor/optim_world_size_4_rank_3.pt").unlink()
            (root / "latest_checkpointed_iteration.txt").write_text("40")
            with self.assertRaises(RuntimeError):
                publish_last(root)
            self.assertEqual((root / "last").resolve(), old)
            self.assertTrue(old.exists())

    def test_unfinished_marker_does_not_publish_partial_save(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = self.make_checkpoint(root, 20)
            (root / "last").symlink_to(old.name)
            (root / "global_step_40").mkdir()
            (root / "latest_checkpointed_iteration.txt").write_text("")
            self.assertIsNone(publish_last(root))
            self.assertEqual((root / "last").resolve(), old)

    def test_external_symlink_is_never_followed_or_deleted(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            tempfile.TemporaryDirectory() as other,
        ):
            root = Path(directory)
            self.make_checkpoint(root, 20)
            (root / "global_step_1").symlink_to(other)
            (root / "latest_checkpointed_iteration.txt").write_text("20")
            publish_last(root)
            self.assertTrue((root / "global_step_1").is_symlink())
            self.assertTrue(Path(other).is_dir())
            (root / "last").unlink()
            (root / "last").symlink_to(other)
            with self.assertRaises(RuntimeError):
                publish_last(root)

    def test_config_uses_native_verl_and_disjoint_validation(self):
        cfg = build_config(Path("/tmp/run"), Path("/tmp/checkpoints"), 29)
        self.assertEqual(cfg.trainer.total_training_steps, 300)
        self.assertEqual(cfg.trainer.save_freq, 20)
        self.assertEqual(cfg.trainer.test_freq, 20)
        self.assertIsNone(cfg.trainer.max_actor_ckpt_to_keep)
        self.assertNotEqual(cfg.data.train_files, cfg.data.val_files)
        self.assertEqual(
            cfg.data.train_batch_size * cfg.actor_rollout_ref.rollout.n, 32
        )
        self.assertEqual(
            32 // (4 * cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu), 8
        )
        self.assertEqual(
            cfg.data.max_prompt_length + cfg.data.max_response_length, 1280
        )
        self.assertEqual(cfg.actor_rollout_ref.actor.fsdp_config.seed, 29)
        self.assertEqual(
            cfg.actor_rollout_ref.actor.optim.lr_scheduler_type, "constant"
        )
        self.assertEqual(cfg.actor_rollout_ref.rollout.val_kwargs.n, 1)
        self.assertFalse(cfg.actor_rollout_ref.rollout.val_kwargs.do_sample)

    def test_overfitting_inherits_native_config_and_reaches_full_target(self):
        path = Path(__file__).resolve().parents[1] / "configs/verl_math_overfitting.yaml"
        settings = load_settings(path)
        cfg = build_config(Path("/tmp/overfitting-run"), settings=settings)
        self.assertEqual(cfg.data.train_max_samples, 4)
        self.assertEqual(cfg.trainer.total_training_steps, 10000)
        self.assertEqual(cfg.trainer.total_rollout, 320000)
        self.assertGreaterEqual(
            (cfg.data.train_max_samples // cfg.data.train_batch_size)
            * cfg.trainer.total_epochs,
            cfg.trainer.total_training_steps,
        )
        settings.trainer.total_training_steps = 123
        settings.data.rollout_per_question = 4
        updated = build_config(Path("/tmp/overfitting-run"), settings=settings)
        self.assertEqual(updated.trainer.total_rollout, 123 * 4 * 4)
        self.assertEqual(updated.trainer.total_epochs, 123)

    def test_yaml_changes_propagate_and_saved_config_is_frozen(self):
        settings = load_settings()
        settings.launcher.seed = 29
        settings.launcher.gpu_ids = [2, 3]
        settings.launcher.wandb.mode = "disabled"
        settings.data.max_response_length = 1024
        settings.data.train_batch_size = 8
        settings.trainer.total_training_steps = 120
        settings.trainer.save_freq = 10
        cfg = build_config(Path("/tmp/run"), settings=settings)
        self.assertEqual(cfg.trainer.total_training_steps, 120)
        self.assertEqual(cfg.actor_rollout_ref.actor.optim.total_training_steps, 120)
        self.assertEqual(cfg.actor_rollout_ref.actor.ppo_mini_batch_size, 8)
        self.assertEqual(cfg.actor_rollout_ref.rollout.max_model_len, 1280)
        self.assertEqual(cfg.actor_rollout_ref.rollout.max_num_batched_tokens, 1280)
        self.assertEqual(cfg.trainer.n_gpus_per_node, 2)
        self.assertEqual(cfg.actor_rollout_ref.rollout.seed, 29)
        self.assertEqual(cfg.trainer.logger, ["file"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            OmegaConf.save(cfg, path)
            settings.trainer.total_training_steps = 400
            with patch.dict("os.environ", {"WANDB_MODE": "online"}):
                saved = OmegaConf.load(path)
                self.assertEqual(saved.trainer.total_training_steps, 120)
                self.assertEqual(saved.launcher.wandb.mode, "disabled")

    def test_typo_and_inconsistent_context_or_batch_fail_before_training(self):
        settings = load_settings()
        with self.assertRaises(ConfigAttributeError):
            settings.actor_rollout_ref.actor.optim.learning_rat = 1e-5
        for field, value in (
            ("actor_rollout_ref.rollout.max_model_len", 512),
            ("actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu", 3),
            ("launcher.gpu_ids", [0, 1, 2, 3]),
        ):
            with self.subTest(field=field):
                settings = load_settings()
                OmegaConf.update(settings, field, value)
                with self.assertRaises(ValueError):
                    build_config(Path("/tmp/run"), settings=settings)

    def test_two_gpu_checkpoint_and_keep_history_option(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = self.make_checkpoint(root, 10, world_size=2)
            new = self.make_checkpoint(root, 20, world_size=2)
            (root / "last").symlink_to(old.name)
            (root / "latest_checkpointed_iteration.txt").write_text("20")
            with self.assertRaises(RuntimeError):
                publish_last(root, world_size=4)
            self.assertEqual(publish_last(root, world_size=2, keep_last_only=False), 20)
            self.assertEqual((root / "last").resolve(), new)
            self.assertTrue(old.exists())

    def test_reward_options_change_score_but_not_correctness(self):
        cfg = build_config(Path("/tmp/run"))
        options = dict(cfg.reward.custom_reward_function.reward_kwargs)
        options.update(correctness_weight=7.0, format_weight=0.0, numeric_weight=0.0)
        output = SimpleNamespace(returncode=0, stdout='[{"correct": true}]')
        with patch("train_verl.subprocess.run", return_value=output) as grader:
            score = compute_score("math500", "<answer>9</answer>", "9", **options)
        self.assertEqual(score["score"], 7.0)
        self.assertEqual(score["acc"], 1.0)
        self.assertEqual(grader.call_args.args[0][0], options["grader_python"])
        self.assertEqual(grader.call_args.kwargs["timeout"], options["grader_timeout"])

    def test_resume_uses_saved_update_target_without_reading_current_yaml(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            root = run / "checkpoints"
            self.make_checkpoint(root, 120)
            (root / "latest_checkpointed_iteration.txt").write_text("120")
            cfg = build_config(run, root)
            cfg.trainer.total_training_steps = 120
            cfg.actor_rollout_ref.actor.optim.total_training_steps = 120
            OmegaConf.save(cfg, run / "config.yaml")
            manifest = {"checkpoints": str(root), "parquet_sha256": {}}
            with (
                patch.object(
                    sys, "argv", ["train_verl.py", "resume", "--run", str(run)]
                ),
                patch(
                    "train_verl.load_settings",
                    side_effect=AssertionError("must use saved config"),
                ),
                patch("train_verl.read_manifest", return_value=manifest),
                patch("train_verl.RUNS", run),
                patch("train_verl.execute") as execute,
                patch("builtins.print") as output,
                patch.dict("os.environ", {}),
            ):
                main()
            execute.assert_not_called()
            self.assertIn("120 updates", output.call_args.args[0])


class HydraCliTest(unittest.TestCase):
    repo = Path(__file__).resolve().parents[1]

    def cli(self, *arguments, bootstrap=None):
        entry = ["main.py"] if bootstrap is None else ["-c", bootstrap]
        return subprocess.run(
            [sys.executable, *entry, *arguments],
            cwd=self.repo,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "WANDB_MODE": "disabled"},
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    def test_native_config_selection_and_overrides_without_running_a_job(self):
        result = self.cli(
            "--config-name",
            "verl_math",
            "--cfg",
            "job",
            "--resolve",
            "trainer.total_training_steps=120",
            "data.max_response_length=1024",
            "launcher.gpu_ids=[2,3]",
            "launcher.seed=29",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        cfg = OmegaConf.create(result.stdout)
        self.assertEqual(cfg.trainer.total_training_steps, 120)
        self.assertEqual(cfg.actor_rollout_ref.actor.optim.total_training_steps, 120)
        self.assertEqual(cfg.actor_rollout_ref.rollout.max_model_len, 1280)
        self.assertEqual(cfg.trainer.n_gpus_per_node, 2)
        self.assertEqual(cfg.actor_rollout_ref.actor.fsdp_config.seed, 29)

    def test_cli_rejects_typos_and_resume_training_overrides(self):
        result = self.cli("--cfg", "job", "trainer.total_train_steps=120")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("total_train_steps", result.stderr)
        result = self.cli(
            "launcher.command=resume", "actor_rollout_ref.actor.optim.lr=1e-5"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("새 학습 옵션을 제거하세요", result.stderr)

    def test_hydra_multirun_composes_each_seed_without_nested_initialization(self):
        # Replace only the lifecycle call. Hydra's real sweeper and build_config
        # still run, but no data/model/GPU worker is loaded in this contract test.
        bootstrap = """
import sys
from pathlib import Path
sys.path.insert(0, "script")
import train_verl
def inspect(command, settings, selected_run):
    cfg = train_verl.build_config(Path("/tmp/hydra-test-run"), settings=settings)
    print("SWEEP_SEED", cfg.launcher.seed, cfg.actor_rollout_ref.rollout.seed)
train_verl.run_command = inspect
train_verl.main()
"""
        with tempfile.TemporaryDirectory() as directory:
            result = self.cli(
                "--multirun",
                "launcher.command=check",
                "launcher.seed=17,29",
                f"hydra.sweep.dir={directory}",
                bootstrap=bootstrap,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("SWEEP_SEED 17 17", result.stdout)
        self.assertIn("SWEEP_SEED 29 29", result.stdout)


if __name__ == "__main__":
    unittest.main()
