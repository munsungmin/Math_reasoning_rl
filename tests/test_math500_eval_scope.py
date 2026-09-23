"""Check that the live run's final evaluator expands to all 500 exactly once."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'script'))
from math500_eval_scope import LEVEL_COUNTS, publish_results, read_rows, resolve_scope, write_json


class EvaluationScopeTest(unittest.TestCase):
    def fixture(self, root):
        data = root / 'full_data'
        data.mkdir()
        prompts, records = [], []
        for level, count in LEVEL_COUNTS.items():
            for _ in range(count):
                i = str(len(prompts))
                prompts.append(dict(sample_id=i, prompt=f'Question {i}', answer=i))
                records.append(dict(sample_id=i, level=level, subject='Algebra'))
        (data / 'prompts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in prompts))
        manifest = dict(records=records, source_sha256='test-source',
                        prompts_sha256=hashlib.sha256((data/'prompts.jsonl').read_bytes()).hexdigest())
        write_json(data / 'manifest.json', manifest)
        subset = root / 'old_prompts.jsonl'
        subset.write_text(''.join(json.dumps(r)+'\n' for r in prompts[133:]))
        request = dict(version=1, scope='math500_all_levels', data_dir=str(data),
                       manifest_sha256=hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest(),
                       evaluation_outputs=['evaluation_last', 'evaluation_best_validation'],
                       split_ids={'train': [str(i) for i in range(133,397)],
                                  'validation': [str(i) for i in range(397,427)],
                                  'test': [str(i) for i in range(427,464)]})
        write_json(root / 'final_evaluation_request.json', request)
        args = SimpleNamespace(out=root/'evaluation_last', prompts=subset, expected_prompts=367,
                               n=1, temperature=0, seed=17137, max_new_tokens=1024)
        return args, prompts

    def test_explicit_request_changes_only_future_evaluation_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args, _ = self.fixture(root)
            original = args.prompts.read_bytes()
            scope = resolve_scope(args)
            self.assertEqual(args.expected_prompts, 500)
            self.assertEqual(args.out, root/'evaluation_last_math500_all')
            self.assertEqual(len(read_rows(args.prompts)), 500)
            self.assertEqual(len(scope['previous_ids']), 367)
            self.assertEqual((root/'old_prompts.jsonl').read_bytes(), original)
            self.assertFalse(args.out.exists())

    def test_unrequested_evaluations_are_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(out=Path(tmp)/'manual')
            self.assertIsNone(resolve_scope(args))
            args, _ = self.fixture(Path(tmp))
            args.out = Path(tmp)/'manual'
            self.assertIsNone(resolve_scope(args))

    def test_changed_original_prompt_and_corrupted_full_input_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _ = self.fixture(Path(tmp))
            original = args.prompts.read_text()
            args.prompts.write_text(original.replace('Question 133', 'changed question', 1))
            with self.assertRaises(AssertionError):
                resolve_scope(args)
            args.prompts.write_text(original)
            full = Path(tmp)/'full_data/prompts.jsonl'
            full.write_text(full.read_text() + '\n')
            with self.assertRaises(AssertionError):
                resolve_scope(args)

    def test_publish_full_500_and_exact_367_view_without_extra_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args, prompts = self.fixture(root)
            scope = resolve_scope(args)
            train = set(scope['request']['split_ids']['train'])
            rows = [dict(sample_id=p['sample_id'], input=p['prompt'], answer=p['answer'],
                         output='a generated answer', correct=p['sample_id'] in train,
                         hit_length_limit=False, extraction_status='explicit') for p in prompts]
            summary = dict(model='base', adapter='last', adapter_sha256='a', grader_sha256='g',
                           temperature=0, seed=17137, max_new_tokens=1024, n_per_problem=1,
                           correct=264, samples=500, accuracy=264/500, elapsed_seconds=10,
                           per_problem={p['sample_id']: {} for p in prompts},
                           prompt_sha256={p['sample_id']: 'hash' for p in prompts})
            args.out.mkdir()
            publish_results(args.out, rows, summary, scope)
            report = json.loads((args.out/'benchmark_summary.json').read_text())
            self.assertEqual(report['all_500']['samples'], 500)
            self.assertEqual(report['all_500']['correct'], 264)
            self.assertEqual(report['outside_training_split']['samples'], 236)
            self.assertEqual(report['outside_training_split']['correct'], 0)
            self.assertEqual(report['splits']['test']['samples'], 37)
            self.assertEqual({int(k):v['samples'] for k,v in report['by_level'].items()}, LEVEL_COUNTS)
            legacy = root/'evaluation_last'
            self.assertEqual(len(read_rows(legacy/'answers.jsonl')), 367)
            self.assertEqual(json.loads((legacy/'summary.json').read_text())['samples'], 367)
            self.assertEqual(json.loads((root/'math500_all_results.json').read_text())['results']['evaluation_last']['all_500']['samples'], 500)
            # The second selected model appends its own full scores to the same report.
            scope['compatibility_output'] = str(root/'evaluation_best_validation')
            other = root/'evaluation_best_validation_math500_all'
            other.mkdir()
            publish_results(other, rows, {**summary, 'adapter':'best'}, scope)
            index = json.loads((root/'math500_all_results.json').read_text())
            self.assertEqual(len(index['results']), 2)
            self.assertIn('전체 500', (root/'math500_all_report_ko.md').read_text())


if __name__ == '__main__':
    unittest.main()
