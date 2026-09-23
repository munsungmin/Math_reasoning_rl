"""Lightning LoRA C/S stage with masked targets and shared stage exports."""

import os
import json
import time
import fcntl
import hashlib
import math
import random
from pathlib import Path

launch_rank = int(os.environ.get("LOCAL_RANK", 0))
physical = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[launch_rank]
os.environ["CUDA_VISIBLE_DEVICES"] = physical
os.environ["LOCAL_RANK"] = "0"
os.environ["GROUP_RANK"] = os.environ["RANK"]
os.environ["LOCAL_WORLD_SIZE"] = "1"
import torch
import numpy as np

STAGE = Path(os.environ["STAGE_ROOT"])
OUT = Path(os.environ["STAGE_ARTIFACT"])
OUT.mkdir(parents=True, exist_ok=True)
rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
with (OUT / "cuda_init.lock").open("w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    torch.cuda.set_device(0)
    torch.cuda.synchronize()
    fcntl.flock(lock, fcntl.LOCK_UN)
(OUT / f"cuda_ready_{rank}").touch()
while len(list(OUT.glob("cuda_ready_*"))) < world:
    time.sleep(0.2)
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch.plugins.precision import MixedPrecision
from lightning.pytorch.loggers import CSVLogger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, PeftModel
from torch.utils.data import Dataset, DataLoader

MODEL = "/data/sungmin/math_reasoning/models/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"
DATA = Path(
    "/data/sungmin/math_reasoning/artifacts/math_error_transfer/frozen/v1_20260910"
)
KIND = os.environ["STAGE_KIND"]
SEED = int(os.environ["STAGE_SEED"])
PARENT = os.environ.get("STAGE_PARENT")
EPOCHS = 2 if KIND == "solve_sft" else 3
pl.seed_everything(SEED, workers=True)
tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
CODES = ["A", "B", "C", "D"] if KIND == "classify_multi" else ["A", "B"]
CODE_IDS = [tok.encode(c, add_special_tokens=False) for c in CODES]
assert all(len(x) == 1 for x in CODE_IDS)
CODE_IDS = [x[0] for x in CODE_IDS]


class Rows(Dataset):
    def __init__(self, split):
        path = DATA / (
            "sft_train.jsonl" if KIND == "solve_sft" else f"perl_{split}.jsonl"
        )
        self.rows = [json.loads(l) for l in path.read_text().splitlines()]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        if KIND == "solve_sft":
            prompt = r["prompt"]
            target = r["target"]
            label = -1
        else:
            kind = KIND.removeprefix("classify_")
            prompt = r["prompts"][kind]
            label = (
                {
                    "Correct": 0,
                    "Mathematical_Error": 1,
                    "Logical_Inconsistency": 2,
                    "Accumulation_Error": 3,
                }[r["label"]]
                if kind == "multi"
                else (
                    int(r["label"] != "Correct")
                    if kind == "binary"
                    else int(r["label"] == "Logical_Inconsistency")
                )
            )
            target = CODES[label]
        p = tok.encode(prompt, add_special_tokens=False)
        t = tok.encode(target, add_special_tokens=False) + [tok.eos_token_id]
        assert len(p) + len(t) <= 2048
        return dict(
            id=r["id"],
            input_ids=torch.tensor(p + t),
            target=torch.tensor(t),
            label=label,
        )


def collate(rows):
    assert len(rows) == 1
    r = rows[0]
    return dict(
        id=r["id"],
        input_ids=r["input_ids"][None],
        target=r["target"][None],
        label=r["label"],
    )


train_data = Rows("train")
valid_data = Rows("validation") if KIND != "solve_sft" else None


class Task(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        if PARENT:
            self.model = PeftModel.from_pretrained(
                self.model, PARENT, is_trainable=True
            )
        else:
            self.model = get_peft_model(
                self.model,
                LoraConfig(
                    r=16,
                    lora_alpha=32,
                    lora_dropout=0.05,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    task_type="CAUSAL_LM",
                ),
            )
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        self.model.enable_input_require_grads()
        self.model.config.use_cache = False
        self.epoch_losses = []
        self.validation_rows = []
        self.history = []
        self.validation_history = []

    def target_logits(self, batch):
        # Only project positions that predict target tokens; prompt loss stays masked.
        count = batch["target"].shape[1]
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=torch.ones_like(batch["input_ids"]),
            logits_to_keep=count + 1,
        ).logits[:, :-1]

    def training_step(self, batch, batch_idx):
        logits = self.target_logits(batch)
        loss = torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            batch["target"].reshape(-1),
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite supervised loss")
        self.epoch_losses.append(loss.detach())
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=1,
        )
        return loss

    def on_before_optimizer_step(self, optimizer):
        if any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for p in self.model.parameters()
            if p.requires_grad
        ):
            raise RuntimeError("Non-finite supervised gradients")

    def on_train_epoch_end(self):
        value = torch.stack(self.epoch_losses).mean()
        torch.distributed.all_reduce(value)
        value /= world
        self.history.append(
            dict(epoch=int(self.current_epoch) + 1, loss=float(value))
        )
        self.epoch_losses = []
        if rank == 0:
            (OUT / "loss_history.json").write_text(
                json.dumps(self.history, indent=2)
            )

    def validation_step(self, batch, batch_idx):
        logits = self.target_logits(batch)[:, 0, CODE_IDS]
        pred = int(logits.argmax(-1))
        self.validation_rows.append(
            dict(id=batch["id"], gold=batch["label"], prediction=pred)
        )

    def on_validation_epoch_end(self):
        shards = [None] * world
        torch.distributed.all_gather_object(shards, self.validation_rows)
        self.validation_rows = []
        unique = {r["id"]: r for shard in shards for r in shard}
        cm = np.zeros((len(CODES), len(CODES)), dtype=int)
        for r in unique.values():
            cm[r["gold"], r["prediction"]] += 1
        tp = cm.diagonal()
        f1 = 2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1)
        majority = int(cm.sum(1).argmax())
        majority_f1 = (
            2 * cm.sum(1)[majority] / (cm.sum() + cm.sum(1)[majority])
        ) / len(CODES)
        metrics = dict(
            epoch=int(self.current_epoch) + 1,
            accuracy=float(tp.sum() / cm.sum()),
            macro_f1=float(f1.mean()),
            majority_macro_f1=float(majority_f1),
            confusion_matrix=cm.tolist(),
            examples=len(unique),
        )
        self.validation_history.append(metrics)
        if rank == 0:
            (OUT / "classification_metrics.json").write_text(
                json.dumps(self.validation_history, indent=2)
            )
            (
                OUT / f"validation_epoch{self.current_epoch + 1}.jsonl"
            ).write_text(
                "".join(json.dumps(r) + "\n" for r in unique.values())
            )

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=1e-5,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )
        total = math.ceil(math.ceil(len(train_data) / world) / 8) * EPOCHS
        sch = get_linear_schedule_with_warmup(opt, round(total * 0.05), total)
        return dict(
            optimizer=opt, lr_scheduler=dict(scheduler=sch, interval="step")
        )

    def on_save_checkpoint(self, checkpoint):
        state = dict(
            python=random.getstate(),
            numpy=np.random.get_state(),
            torch=torch.get_rng_state(),
            cuda=torch.cuda.get_rng_state(),
        )
        states = [None] * world
        torch.distributed.all_gather_object(states, state)
        checkpoint["rank_rng_states"] = states
        checkpoint["stage_loss_history"] = self.history
        checkpoint["stage_validation_history"] = self.validation_history

    def on_load_checkpoint(self, checkpoint):
        state = checkpoint["rank_rng_states"][rank]
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"].cpu())
        torch.cuda.set_rng_state(state["cuda"].cpu())
        self.history = checkpoint["stage_loss_history"]
        self.validation_history = checkpoint["stage_validation_history"]


class CompactState(Callback):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["state_dict"] = {
            n: t for n, t in checkpoint["state_dict"].items() if "lora_" in n
        }


# strict_loading=False permits frozen base tensors omitted from resume.
task = Task()
task.strict_loading = False
checkpoint = ModelCheckpoint(
    dirpath=STAGE / "resume",
    filename="epoch-{epoch:02d}",
    every_n_epochs=1,
    save_top_k=-1,
    save_last=True,
)
trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    num_nodes=world,
    strategy="ddp_find_unused_parameters_false",
    max_epochs=EPOCHS,
    accumulate_grad_batches=8,
    gradient_clip_val=1.0,
    plugins=[
        MixedPrecision(
            "16-mixed",
            device="cuda",
            scaler=torch.amp.GradScaler("cuda", init_scale=1024),
        )
    ],
    callbacks=[checkpoint, CompactState()],
    logger=CSVLogger(str(OUT), name="metrics"),
    num_sanity_val_steps=0,
    log_every_n_steps=1,
    enable_progress_bar=False,
    default_root_dir=STAGE,
)
train_loader = DataLoader(
    train_data, batch_size=1, shuffle=True, collate_fn=collate, num_workers=0
)
valid_loader = (
    DataLoader(valid_data, batch_size=1, collate_fn=collate, num_workers=0)
    if valid_data
    else None
)
trainer.fit(
    task,
    train_loader,
    valid_loader,
    ckpt_path=os.environ.get("STAGE_RESUME") or None,
)
export = STAGE / "exports/final"
if rank == 0:
    task.model.save_pretrained(export)
    tok.save_pretrained(export)
    cpu = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, local_files_only=True
    )
    check = PeftModel.from_pretrained(cpu, export, is_trainable=True)
    before = {
        n: p.detach().cpu()
        for n, p in task.model.named_parameters()
        if p.requires_grad
    }
    after = {
        n: p.detach().cpu()
        for n, p in check.named_parameters()
        if p.requires_grad
    }
    assert set(before) == set(after) and all(
        torch.equal(p, after[n]) for n, p in before.items()
    )
    # Read the native checkpoint and verify the independent resume payload.
    saved = torch.load(
        checkpoint.last_model_path, map_location="cpu", weights_only=False
    )
    for key in [
        "state_dict",
        "optimizer_states",
        "lr_schedulers",
        "loops",
        "rank_rng_states",
    ]:
        assert key in saved, key
    assert len(saved["rank_rng_states"]) == world
    passed = task.history[-1]["loss"] < task.history[0]["loss"]
    if valid_data:
        passed = (
            passed
            and task.validation_history[-1]["macro_f1"]
            > task.validation_history[-1]["majority_macro_f1"]
        )
    summary = dict(
        completed=True,
        epochs=EPOCHS,
        global_step=trainer.global_step,
        reload_verified=True,
        native_resume_payload_verified=True,
        next_stage_gate=passed,
        kind=KIND,
        seed=SEED,
        parent=PARENT,
        optimizer_reset=True,
        train_rows=len(train_data),
        history=task.history,
        validation=task.validation_history,
        precision="fp16 LoRA; shared fixed base",
        classification_logits="restricted to equal-length A/B/C/D code tokens",
    )
    (STAGE / "checkpoint_manifest.json").write_text(
        json.dumps(summary, indent=2)
    )
    print("SUPERVISED_STAGE_RESULT", json.dumps(summary), flush=True)
trainer.strategy.barrier()
