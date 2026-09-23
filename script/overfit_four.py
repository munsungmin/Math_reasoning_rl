"""Supervised recovery and generation evaluation on a frozen four-problem set.

This is explicitly SFT, not GRPO. Gold solutions occur only in training targets;
generation evaluation uses the original prompts and the unchanged MATH grader.
Run with the verl-current Python and an explicit CUDA_VISIBLE_DEVICES value.
"""

import argparse
import hashlib
import json
import random
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

REPO = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def grade(rows, cfg):
    result = subprocess.run(
        [cfg["grader_python"], str(REPO / "script/limit_rlvr_grade.py")],
        input=json.dumps([
            {"dataset": "math500_level3_5", "completion": r["output"], "answer": r["answer"]}
            for r in rows
        ]), text=True, capture_output=True, timeout=180, check=True,
    )
    for row, judgement in zip(rows, json.loads(result.stdout), strict=True):
        row.update(judgement)
    return rows


@torch.inference_mode()
def evaluate(model, tok, rows, cfg, out, label, n=1, temperature=0.0, seed=17001):
    set_seed(seed)
    model.eval()
    model.config.use_cache = True
    prompts = [row for _ in range(n) for row in rows]
    results = []
    started = time.monotonic()
    for start in range(0, len(prompts), cfg["eval_batch_size"]):
        batch = prompts[start:start + cfg["eval_batch_size"]]
        inputs = tok([r["prompt"] for r in batch], padding=True, return_tensors="pt",
                     add_special_tokens=False).to(model.device)
        kwargs = dict(max_new_tokens=cfg["max_new_tokens"], do_sample=temperature > 0,
                      pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                      use_cache=True)
        if temperature > 0:
            kwargs.update(temperature=temperature, top_p=1.0, top_k=0)
        generated = model.generate(**inputs, **kwargs)[:, inputs.input_ids.shape[1]:]
        for row, ids in zip(batch, generated, strict=True):
            ids = ids.tolist()
            ended = tok.eos_token_id in ids
            length = ids.index(tok.eos_token_id) + 1 if ended else len(ids)
            results.append(dict(sample_id=row["sample_id"], input=row["prompt"],
                                answer=row["answer"], output=tok.decode(ids, skip_special_tokens=True),
                                generated_tokens=length, hit_length_limit=not ended,
                                temperature=temperature, seed=seed))
        print(f"{label}: generated {len(results)}/{len(prompts)}", flush=True)
    grade(results, cfg)
    groups = defaultdict(list)
    for row in results:
        groups[row["sample_id"]].append(row)
    summary = dict(label=label, seed=seed, temperature=temperature, n_per_problem=n,
                   samples=len(results), correct=sum(r["correct"] for r in results),
                   accuracy=sum(r["correct"] for r in results) / len(results),
                   length_limit_rate=sum(r["hit_length_limit"] for r in results) / len(results),
                   elapsed_seconds=time.monotonic() - started,
                   per_problem={key: dict(correct=sum(r["correct"] for r in rs), samples=len(rs),
                                          accuracy=sum(r["correct"] for r in rs) / len(rs))
                                for key, rs in groups.items()})
    (out / f"{label}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results))
    write_json(out / f"{label}.summary.json", summary)
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "eval"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=17001)
    parser.add_argument("--label", default="evaluation")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.command == "train" and args.out.exists():
        raise FileExistsError(f"Refusing to overwrite a training run: {args.out}")
    args.out.mkdir(parents=True, exist_ok=args.command == "eval")
    rows = [json.loads(s) for s in Path(cfg["train_rows"]).read_text().splitlines()]
    assert len(rows) == 4 and len({r["sample_id"] for r in rows}) == 4
    set_seed(cfg["seed"])
    torch.set_num_threads(2)
    tok = AutoTokenizer.from_pretrained(cfg["model"], local_files_only=True, padding_side="left")
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=torch.float16,
                                               attn_implementation="sdpa", local_files_only=True)
    adapter = str(args.adapter or cfg["initial_adapter"])
    model = PeftModel.from_pretrained(model, adapter, is_trainable=args.command == "train").cuda()
    if args.command == "eval":
        evaluate(model, tok, rows, cfg, args.out, args.label, args.n, args.temperature, args.seed)
        return
    write_json(args.out / "config.json", cfg)
    write_json(args.out / "provenance.json", dict(
        method="supervised fine tuning on gold solutions of the same four training problems",
        input_adapter=adapter, train_rows_sha256=hashlib.sha256(Path(cfg["train_rows"]).read_bytes()).hexdigest(),
        prompt_sha256={r["sample_id"]: hashlib.sha256(r["prompt"].encode()).hexdigest() for r in rows},
        grader_sha256=hashlib.sha256((REPO / "script/limit_rlvr_grade.py").read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    targets = []
    for row in rows:
        target = "<reasoning>\n" + row["solution"].strip() + "\n</reasoning>\n<answer>" + row["answer"] + "</answer>"
        p = tok.encode(row["prompt"], add_special_tokens=False)
        t = tok.encode(target, add_special_tokens=False) + [tok.eos_token_id]
        assert len(p) + len(t) <= cfg["max_train_tokens"]
        targets.append(dict(sample_id=row["sample_id"], input=row["prompt"], output=target,
                            answer=row["answer"], prompt_tokens=len(p), target_tokens=len(t)))
    grade(targets, cfg)
    assert all(r["correct"] for r in targets), "A supervised target failed the unchanged grader"
    write_json(args.out / "targets.json", targets)
    encoded = []
    for row in targets:
        p = tok.encode(row["input"], add_special_tokens=False)
        t = tok.encode(row["output"], add_special_tokens=False) + [tok.eos_token_id]
        encoded.append((torch.tensor([p + t], device="cuda"), torch.tensor([t], device="cuda")))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    successes = 0
    started = time.monotonic()
    with (args.out / "metrics.jsonl").open("w", buffering=1) as metrics:
        for step in range(1, cfg["max_steps"] + 1):
            model.train()
            model.config.use_cache = False
            optimizer.zero_grad(set_to_none=True)
            indices = list(range(4))
            random.shuffle(indices)
            losses = []
            for index in indices:
                ids, target = encoded[index]
                count = target.shape[1]
                with torch.autocast("cuda", dtype=torch.float16):
                    # Project only positions predicting the response. No prompt loss.
                    logits = model(input_ids=ids, logits_to_keep=count + 1, use_cache=False).logits[:, :-1]
                    loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), target.reshape(-1))
                scaler.scale(loss / 4).backward()
                losses.append(float(loss.detach()))
                del logits, loss
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"], error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            event = dict(step=step, loss=sum(losses) / 4, grad_norm=float(grad_norm),
                         learning_rate=cfg["learning_rate"], elapsed_seconds=time.monotonic() - started)
            metrics.write(json.dumps(event) + "\n")
            print(json.dumps(event), flush=True)
            if step % cfg["eval_every"] == 0 or step == cfg["max_steps"]:
                checkpoint = args.out / f"step_{step:04d}"
                model.save_pretrained(checkpoint)
                tok.save_pretrained(checkpoint)
                summary = evaluate(model, tok, rows, cfg, args.out, f"greedy_step_{step:04d}")
                successes = successes + 1 if summary["correct"] == 4 else 0
                write_json(args.out / "latest.json", dict(step=step, adapter=str(checkpoint), greedy=summary))
                if step >= cfg.get("min_steps", 0) and successes >= cfg["consecutive_greedy_successes"]:
                    print("Stopping after consecutive greedy 4/4; fresh sampled evaluation is still required.", flush=True)
                    break


if __name__ == "__main__":
    main()
