"""Only SFT depends on Lightning. Model, loss, optimizer and observations are injected."""

import random

import lightning.pytorch as pl
import numpy as np
import torch

from scripts.eval.inference.classification import label_token_ids
from scripts.utils.metric import build_metrics


class SupervisedTask(pl.LightningModule):
    def __init__(
        self,
        model,
        tokenizer,
        loss,
        optimizer,
        scheduler,
        train,
        kind="solve",
        labels=(),
        val_metrics=None,
        test_metrics=None,
        sampling=None,
        grader=None,
    ):
        super().__init__()
        self.model, self.tokenizer, self.objective = model, tokenizer, loss
        self.optimizer_factory, self.scheduler_config, self.train_config = (
            optimizer,
            scheduler,
            train,
        )
        self.kind, self.labels = kind, list(labels)
        self.code_ids = (
            label_token_ids(tokenizer, self.labels) if kind == "classify" else None
        )
        self.validation_metrics = build_metrics(val_metrics)
        self.sampling, self.grader = sampling, grader
        self.validation_records = []
        self.validation_loss_sum, self.validation_tokens = 0.0, 0
        self.consumed_ids = []
        self._rank_rng = None

    def forward(self, **inputs):
        return self.model(**inputs)

    def on_train_epoch_start(self):
        self.trainer.datamodule.sampler.set_epoch(self.current_epoch)

    def _restore_training_rng(self):
        if self._rank_rng:
            state = self._rank_rng[self.global_rank]
            random.setstate(state["python"])
            np.random.set_state(state["numpy"])
            torch.set_rng_state(state["torch"].cpu())
            if torch.cuda.is_initialized() and state["cuda"] is not None:
                torch.cuda.set_rng_state(state["cuda"].cpu())
            self._rank_rng = None

    def training_step(self, batch, batch_idx):
        # Loader iterator creation/Lightning setup must finish before restoring dropout RNG.
        self._restore_training_rng()
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )
        terms = self.objective(outputs.logits, batch["labels"])
        if not torch.isfinite(terms["loss"]):
            raise RuntimeError("Non-finite supervised loss")
        self.consumed_ids.extend(batch["sample_id"])
        self.log(
            "train/loss",
            terms["loss"],
            batch_size=terms["target_tokens"],
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        return terms["loss"]

    def validation_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )
        terms = self.objective(outputs.logits, batch["labels"])
        records = [
            {
                "sample_id": identity,
                "loss_sum": float(loss),
                "target_tokens": int(count),
            }
            for identity, loss, count in zip(
                batch["sample_id"],
                terms["sample_loss_sums"],
                terms["sample_token_counts"],
                strict=True,
            )
        ]
        if self.kind == "classify":
            positions = batch["prompt_length"] - 1
            scores = outputs.logits[
                torch.arange(len(positions), device=positions.device), positions
            ][:, self.code_ids]
            for record, label, score in zip(
                records, batch["label"], scores, strict=True
            ):
                record.update(
                    label=int(label),
                    prediction=int(score.argmax()),
                    log_probs=score.float().log_softmax(-1).tolist(),
                )
        self.validation_records.extend(records)

    def on_validation_epoch_end(self):
        records = self.validation_records
        if torch.distributed.is_initialized():
            gathered = [None] * self.trainer.world_size
            torch.distributed.all_gather_object(gathered, records)
            records = [row for shard in gathered for row in shard]
        # DistributedSampler may pad validation. Count each stable identity once.
        records = list({r["sample_id"]: r for r in records}.values())
        tokens = sum(r["target_tokens"] for r in records)
        if tokens:
            self.log(
                "val/loss",
                sum(r["loss_sum"] for r in records) / tokens,
                sync_dist=False,
            )
        if self.kind == "classify":
            self._record_validation(records)
        else:
            self._generate_validation()
        self.validation_records = []
        self.validation_loss_sum, self.validation_tokens = 0.0, 0

    def _record_validation(self, records):
        self.validation_metrics.update(records)
        values = self.validation_metrics.compute()
        for key, value in values.items():
            if isinstance(value, (float, int)):
                self.log(f"val/{key}", float(value), sync_dist=False)
        if self.trainer.is_global_zero:
            from pathlib import Path

            from scripts.utils.io import write_json, write_rows

            directory = (
                Path(self.trainer.default_root_dir) / "val" / f"step_{self.global_step}"
            )
            write_rows(directory / "predictions.jsonl", records)
            write_json(directory / "summary.json", values)
        self.validation_metrics.reset()

    def _generate_validation(self):
        rows = self.trainer.datamodule.data.rows("validation")
        if not rows:
            return
        from hydra.utils import instantiate

        from scripts.eval.pipeline import grade_records
        from scripts.reasoning.rollout.generation import generate

        records = None
        if self.trainer.is_global_zero:
            records = list(
                generate(self.model, self.tokenizer, rows, instantiate(self.sampling))
            )
            grade_records(records, instantiate(self.grader))
        if torch.distributed.is_initialized():
            payload = [records]
            torch.distributed.broadcast_object_list(payload, src=0)
            records = payload[0]
        self._record_validation(records)

    def on_train_epoch_end(self):
        if (
            self.kind == "solve"
            and self.trainer.datamodule.val_set is None
            and (self.current_epoch + 1) % self.trainer.check_val_every_n_epoch == 0
        ):
            self._generate_validation()

    def configure_optimizers(self):
        optimizer = self.optimizer_factory(
            params=[p for p in self.parameters() if p.requires_grad]
        )
        for group in optimizer.param_groups:
            if "betas" in group:
                group["betas"] = tuple(group["betas"])
        cfg = self.scheduler_config
        total = self.trainer.estimated_stepping_batches
        warmup = cfg.warmup_steps or round(total * cfg.warmup_ratio)
        from transformers import get_scheduler

        scheduler = get_scheduler(
            cfg.name, optimizer, num_warmup_steps=warmup, num_training_steps=total
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_save_checkpoint(self, checkpoint):
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None,
        }
        states = [state]
        if torch.distributed.is_initialized():
            states = [None] * self.trainer.world_size
            torch.distributed.all_gather_object(states, state)
        checkpoint["rank_rng_states"] = states
        identities = [self.consumed_ids]
        if torch.distributed.is_initialized():
            identities = [None] * self.trainer.world_size
            torch.distributed.all_gather_object(identities, self.consumed_ids)
        checkpoint["rank_consumed_ids"] = identities

    def on_load_checkpoint(self, checkpoint):
        self._rank_rng = checkpoint["rank_rng_states"]
        if len(self._rank_rng) != self.trainer.world_size:
            raise ValueError("SFT resume requires the original world size")
        self.consumed_ids = checkpoint["rank_consumed_ids"][self.global_rank]
