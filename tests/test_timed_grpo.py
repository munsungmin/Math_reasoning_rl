"""Protect saved weights and evaluation boundaries in the unattended RL run."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'script'))
from run_timed_grpo import collapse_detected, complete_jsonl, prune_checkpoints, split_scores
from train_verl import build_config, load_settings


class TimedGrpoTest(unittest.TestCase):
    def checkpoint(self, root, step):
        path = root / f'global_step_{step}'
        actor = path / 'actor'
        (actor / 'huggingface').mkdir(parents=True)
        (path / 'data.pt').write_text('data')
        (actor / 'fsdp_config.json').write_text('{"world_size":4}')
        for name in ('lora_train_meta.json', 'huggingface/config.json', 'huggingface/tokenizer_config.json'):
            (actor / name).write_text('{}')
        for rank in range(4):
            for kind in ('model', 'optim', 'extra_state'):
                (actor / f'{kind}_world_size_4_rank_{rank}.pt').write_text('shard')
        return path

    def test_rotation_retains_best_and_two_latest_and_ignores_unfinished_save(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as other:
            root = Path(directory)
            for step in (10, 20, 30, 40):
                self.checkpoint(root, step)
            (root / 'last').symlink_to('global_step_40')
            (root / 'latest_checkpointed_iteration.txt').write_text('40')
            (root / 'global_step_50').mkdir()
            (root / 'global_step_1').symlink_to(other)
            self.assertEqual(prune_checkpoints(root, best_step=10), [20])
            for step in (10, 30, 40, 50):
                self.assertTrue((root / f'global_step_{step}').is_dir())
            self.assertTrue((root / 'global_step_1').is_symlink())
            (root / 'last').unlink()
            (root / 'last').symlink_to(other)
            with self.assertRaises(RuntimeError):
                prune_checkpoints(root, best_step=10)

    def test_incomplete_latest_aborts_before_removing_older_saves(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in (10, 20, 30):
                self.checkpoint(root, step)
            (root / 'global_step_30/actor/optim_world_size_4_rank_3.pt').unlink()
            (root / 'latest_checkpointed_iteration.txt').write_text('30')
            with self.assertRaises(RuntimeError):
                prune_checkpoints(root, best_step=0)
            self.assertTrue((root / 'global_step_10').is_dir())

    def test_partial_metric_write_is_retried_but_interior_corruption_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'metrics.jsonl'
            path.write_text('{"step":1}\n{"step":')
            self.assertEqual(complete_jsonl(path), [{'step': 1}])
            path.write_text('{"step":1}\n{"step":2}\n')
            self.assertEqual(len(complete_jsonl(path)), 2)
            path.write_text('broken\n{"step":2}\n')
            with self.assertRaises(json.JSONDecodeError):
                complete_jsonl(path)

    def test_collapse_requires_three_large_drops_and_at_least_30_steps(self):
        baseline = [{'step': 0, 'accuracy': .6}]
        self.assertFalse(collapse_detected(baseline + [{'step': i, 'accuracy': .3} for i in (10, 20)]))
        self.assertFalse(collapse_detected(baseline + [{'step': i, 'accuracy': .6} for i in (10, 20, 30)]))
        self.assertFalse(collapse_detected(baseline + [{'step': i, 'accuracy': .3} for i in (1, 2, 3)]))
        self.assertTrue(collapse_detected(baseline + [{'step': i, 'accuracy': .3} for i in (10, 20, 30)]))

    def test_training_correctness_does_not_inflate_test_or_outside_train(self):
        rows = [dict(sample_id=str(i), correct=i < 264, hit_length_limit=False) for i in range(367)]
        ids = {name: [str(i) for i in indices] for name, indices in
               [('train', range(264)), ('validation', range(264, 294)), ('test', range(294, 331))]}
        scores = split_scores(rows, ids)
        self.assertEqual(scores['train']['accuracy'], 1)
        self.assertEqual(scores['test']['accuracy'], 0)
        self.assertEqual(scores['outside_training_split']['samples'], 103)
        self.assertEqual(scores['outside_training_split']['accuracy'], 0)
        with self.assertRaises(AssertionError):
            split_scores(rows[:-1] + rows[:1], ids)

    def test_effective_config_is_pure_grpo_with_all_training_rows_and_deadline_in_ray(self):
        config = Path(__file__).resolve().parents[1] / 'configs/verl_math_full_train_16h.yaml'
        with patch.dict('os.environ', {'MATH_RL_DEADLINE': '2000000000.5'}):
            cfg = build_config(Path('/tmp/timed-grpo-test'), settings=load_settings(config))
        self.assertEqual(cfg.data.train_max_samples, -1)
        self.assertEqual(cfg.data.train_batch_size * cfg.actor_rollout_ref.rollout.n, 128)
        self.assertEqual(cfg.actor_rollout_ref.rollout.n, 16)
        self.assertEqual(cfg.algorithm.adv_estimator, 'grpo')
        self.assertFalse(cfg.distillation.enabled)
        self.assertFalse(cfg.actor_rollout_ref.actor.use_kl_loss)
        self.assertEqual(cfg.actor_rollout_ref.actor.entropy_coeff, 0)
        self.assertEqual(cfg.ray_kwargs.ray_init.runtime_env.env_vars.MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP,
                         '2000000000.5')
        self.assertEqual(cfg.trainer.save_freq, cfg.trainer.test_freq)
        self.assertEqual(cfg.actor_rollout_ref.actor.ppo_epochs, 1)
        self.assertFalse(cfg.launcher.keep_last_only)


if __name__ == '__main__':
    unittest.main()
