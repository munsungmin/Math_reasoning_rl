"""Training handoff ordering, split expansion, and all-levels score boundaries."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'script'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'script/analysis'))
from math500_eval_scope import LEVEL_COUNTS, write_json
from prepare_math500_continuation import take_stratified
import run_math500_continuation as coordinator
import run_timed_grpo as timed
from training_handoff import defer_for_training


class ContinuationTest(unittest.TestCase):
    def fixture(self, root):
        source=root/'source'
        source.mkdir()
        out=root/'next'
        (out/'data').mkdir(parents=True)
        (out/'data/manifest.json').write_text('{}')
        native=source/'native'
        native.mkdir()
        (native/'config.yaml').write_text('algorithm:\n  adv_estimator: grpo\ndistillation:\n  enabled: false\n')
        adapter=source/'export_step_20/lora_adapter'
        adapter.mkdir(parents=True)
        (adapter/'adapter_model.safetensors').write_bytes(b'test weights')
        write_json(source/'training_input_audit.json',dict(no_solution_targets=True,distillation=False))
        write_json(source/'protocol.json',dict(deadline_epoch=2000000000))
        state=dict(status='failed',error='Stage failed with exit 75: evaluation_last.log',
                   saved_last_step=20,checkpoints=str(source/'checkpoints'),run=str(native))
        write_json(source/'status.json',state)
        request=dict(version=1,mode='train_then_evaluate',source_run=str(source),next_run=str(out),
                     source_supervisor_pid=987654321,source_supervisor_start='old-process',hours=16,config='config.yaml')
        path=source/'next_training_request.json'
        write_json(path,request)
        args=SimpleNamespace(out=source/'evaluation_last',adapter=adapter)
        return source,out,path,args

    def test_defer_writes_handoff_without_creating_evaluation_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,out,path,args=self.fixture(Path(tmp))
            with self.assertRaises(SystemExit) as caught:
                defer_for_training(args)
            self.assertEqual(caught.exception.code,75)
            self.assertFalse(args.out.exists())
            handoff=json.loads((source/'training_handoff.json').read_text())
            self.assertEqual(handoff['next_run'],str(out))
            self.assertEqual(handoff['request_sha256'],hashlib.sha256(path.read_bytes()).hexdigest())

    def test_waits_for_old_supervisor_before_starting_next_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,out,path,args=self.fixture(Path(tmp))
            with self.assertRaises(SystemExit):
                defer_for_training(args)
            calls=[]
            def run_stage(command,env,log_path,timeout):
                calls.append('train')
                self.assertIn('--data-manifest',command)
                self.assertEqual(env['MATH_CONTINUATION_ADAPTER'],str(args.adapter))
                self.assertEqual(command[command.index('--hours')+1],'16')
                (out/'training').mkdir()
                write_json(out/'training/status.json',dict(status='complete',results=str(out/'training/results.json')))
            with patch.object(sys,'argv',['coordinator','--request',str(path)]), \
                 patch.object(coordinator,'process_start',side_effect=['old-process',None]), \
                 patch.object(coordinator,'validate_checkpoint',return_value=source/'checkpoints/global_step_20'), \
                 patch.object(coordinator.time,'sleep',side_effect=lambda _:calls.append('wait')), \
                 patch.object(timed,'wait_for_free_gpus',side_effect=lambda _:calls.append('free')), \
                 patch.object(timed,'run_stage',side_effect=run_stage):
                coordinator.main()
            self.assertEqual(calls,['wait','free','train'])
            self.assertEqual(json.loads((source/'status.json').read_text())['status'],'training_complete_evaluation_deferred')
            self.assertEqual(json.loads((out/'status.json').read_text())['status'],'complete')

    def test_unexpected_source_failure_is_not_treated_as_successful_handoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,out,path,args=self.fixture(Path(tmp))
            with self.assertRaises(SystemExit):
                defer_for_training(args)
            state=json.loads((source/'status.json').read_text())
            state['error']='GPU training failed'
            write_json(source/'status.json',state)
            with self.assertRaises(AssertionError):
                coordinator.validate_handoff(source,json.loads(path.read_text()))

    def test_stratified_selection_has_exact_size_and_is_reproducible(self):
        rows=[dict(sample_id=str(i),level=1 if i<43 else 2) for i in range(133)]
        selected,remaining=take_stratified(rows,20,'seed17')
        repeated,_=take_stratified(list(reversed(rows)),20,'seed17')
        self.assertEqual(selected,repeated)
        self.assertEqual(len(selected),20)
        self.assertEqual(len(remaining),113)
        self.assertFalse({r['sample_id'] for r in selected}&{r['sample_id'] for r in remaining})

    def test_all_level_training_score_does_not_inflate_heldout_score(self):
        metadata=[]
        for level,count in LEVEL_COUNTS.items():
            metadata.extend(dict(level=level) for _ in range(count))
        metadata=[dict(sample_id=str(i),level=m['level']) for i,m in enumerate(metadata)]
        rows=[dict(sample_id=str(i),correct=i<400,hit_length_limit=False) for i in range(500)]
        splits={'train':[str(i) for i in range(400)],'validation':[str(i) for i in range(400,450)],'test':[str(i) for i in range(450,500)]}
        scores=timed.split_scores(rows,splits,metadata)
        self.assertEqual(scores['all_500']['accuracy'],.8)
        self.assertEqual(scores['test']['accuracy'],0)
        self.assertEqual(scores['outside_training_split']['samples'],100)
        self.assertEqual(scores['all_levels_3_5']['samples'],367)
        self.assertEqual(scores['level_1']['samples'],43)


if __name__=='__main__':
    unittest.main()
