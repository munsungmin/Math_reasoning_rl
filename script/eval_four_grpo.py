"""Inference-only evaluation of a GRPO adapter on frozen prompts (four by default).

No optimizer, training targets, or gold solutions are used. Final answers are
sent to the unchanged grader only after generation has finished.
"""

import argparse
import hashlib
import json
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from math500_eval_scope import publish_results, resolve_scope, write_json
from training_handoff import defer_for_training


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--prompts', type=Path, required=True)
    parser.add_argument('--grader-python', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--n', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=17101)
    parser.add_argument('--max-new-tokens', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--expected-prompts', type=int, default=4,
                        help='Explicit expected number of distinct frozen evaluation prompts.')
    args = parser.parse_args()
    defer_for_training(args)
    scope = resolve_scope(args)
    args.out.mkdir(parents=True, exist_ok=False)
    if scope:
        write_json(args.out / 'scope.json', {k: v for k, v in scope.items() if k != 'prompts'})
        print('Applying explicit final-evaluation request: all 500 MATH-500 questions, levels 1-5', flush=True)
    rows = [json.loads(s) for s in args.prompts.read_text().splitlines()]
    assert len(rows) == args.expected_prompts and len({r['sample_id'] for r in rows}) == args.expected_prompts
    assert all(set(r) == {'sample_id', 'prompt', 'answer'} for r in rows)
    set_seed(args.seed)
    torch.set_num_threads(2)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True, padding_side='left')
    tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16,
                                               attn_implementation='sdpa', local_files_only=True)
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False).cuda().eval()
    assert not any(p.requires_grad for p in model.parameters())
    model.config.use_cache = True
    started = time.monotonic()
    results = []
    prompts = rows * args.n
    for start in range(0, len(prompts), args.batch_size):
        batch = prompts[start:start + args.batch_size]
        inputs = tok([r['prompt'] for r in batch], padding=True, return_tensors='pt',
                     add_special_tokens=False).to(model.device)
        kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=args.temperature > 0,
                      pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id, use_cache=True)
        if args.temperature > 0:
            kwargs.update(temperature=args.temperature, top_p=1.0, top_k=0)
        generated = model.generate(**inputs, **kwargs)[:, inputs.input_ids.shape[1]:]
        for row, ids in zip(batch, generated, strict=True):
            ids = ids.tolist()
            ended = tok.eos_token_id in ids
            length = ids.index(tok.eos_token_id) + 1 if ended else len(ids)
            results.append(dict(sample_id=row['sample_id'], input=row['prompt'], answer=row['answer'],
                                output=tok.decode(ids, skip_special_tokens=True), generated_tokens=length,
                                hit_length_limit=not ended, seed=args.seed, temperature=args.temperature))
        with (args.out / 'generated_answers.jsonl').open('a') as f:
            for row in results[-len(batch):]:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(f'Generated {len(results)}/{len(prompts)}', flush=True)
    grader = Path(__file__).resolve().parent / 'limit_rlvr_grade.py'
    for start in range(0, len(results), 32):
        batch = results[start:start + 32]
        graded = subprocess.run([str(args.grader_python), str(grader)], text=True, capture_output=True,
                                timeout=30 + 8 * len(batch), check=True, input=json.dumps([
                                    dict(dataset='math500' if scope else 'math500_level3_5', completion=r['output'], answer=r['answer'])
                                    for r in batch]))
        for row, judgement in zip(batch, json.loads(graded.stdout), strict=True):
            row.update(judgement)
        print(f'Graded {start + len(batch)}/{len(results)}', flush=True)
    groups = defaultdict(list)
    for row in results:
        groups[row['sample_id']].append(row)
    summary = dict(method='Inference only; unchanged original prompts and grader',
                   model=str(args.model), adapter=str(args.adapter), seed=args.seed,
                   temperature=args.temperature, n_per_problem=args.n, samples=len(results),
                   correct=sum(r['correct'] for r in results),
                   accuracy=sum(r['correct'] for r in results) / len(results),
                   length_limit_rate=sum(r['hit_length_limit'] for r in results) / len(results),
                   extraction_status_counts=dict(Counter(r['extraction_status'] for r in results)),
                   max_new_tokens=args.max_new_tokens, elapsed_seconds=time.monotonic() - started,
                   per_problem={p: dict(correct=sum(r['correct'] for r in rs), samples=len(rs),
                                        accuracy=sum(r['correct'] for r in rs) / len(rs))
                                for p, rs in groups.items()},
                   grader_sha256=hashlib.sha256(grader.read_bytes()).hexdigest(),
                   adapter_sha256=hashlib.sha256((args.adapter / 'adapter_model.safetensors').read_bytes()).hexdigest(),
                   prompt_sha256={r['sample_id']: hashlib.sha256(r['prompt'].encode()).hexdigest() for r in rows})
    (args.out / 'answers.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in results))
    (args.out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    if scope:
        publish_results(args.out, results, summary, scope)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
