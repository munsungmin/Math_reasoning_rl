"""The Lightning adapter owns loaders; the common data module owns the examples."""

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, DistributedSampler

from scripts.reasoning.data.dataset import SupervisedCollator, SupervisedRows
from scripts.reasoning.data.sampler import ResumableBatchSampler


class ResumableLoader(DataLoader):
    def state_dict(self):
        return self.batch_sampler.state_dict()

    def load_state_dict(self, state):
        self.batch_sampler.load_state_dict(state)


class LightningData(pl.LightningDataModule):
    def __init__(self, data, tokenizer, train_config, seed, workers=0):
        super().__init__()
        self.data, self.tokenizer, self.config = data, tokenizer, train_config
        self.seed, self.workers = seed, workers
        self.pending_state = None

    def setup(self, stage=None):
        if hasattr(self, "train_set"):
            return
        self.data.validate_splits()
        self.train_set = SupervisedRows(
            self.data.rows("train"), self.tokenizer, self.config.max_tokens
        )
        if not len(self.train_set):
            raise ValueError("SFT requires nonempty training data")
        validation = self.data.rows("validation")
        self.val_set = (
            SupervisedRows(validation, self.tokenizer, self.config.max_tokens)
            if validation and all(r.get("target") for r in validation)
            else None
        )
        self.sampler = ResumableBatchSampler(
            len(self.train_set),
            self.config.batch_size,
            self.seed,
            self.trainer.global_rank,
            self.trainer.world_size,
        )
        if self.pending_state is not None:
            self.sampler.load_state_dict(self.pending_state)
            self.pending_state = None

    def train_dataloader(self):
        return ResumableLoader(
            self.train_set,
            batch_sampler=self.sampler,
            num_workers=self.workers,
            generator=torch.Generator().manual_seed(
                self.seed + self.trainer.global_rank
            ),
            collate_fn=SupervisedCollator(self.tokenizer.pad_token_id),
        )

    def val_dataloader(self):
        if self.val_set is None:
            return []
        sampler = DistributedSampler(
            self.val_set,
            num_replicas=self.trainer.world_size,
            rank=self.trainer.global_rank,
            shuffle=False,
        )
        return DataLoader(
            self.val_set,
            batch_size=self.config.batch_size,
            sampler=sampler,
            generator=torch.Generator().manual_seed(
                self.seed + 10000 + self.trainer.global_rank
            ),
            num_workers=self.workers,
            collate_fn=SupervisedCollator(self.tokenizer.pad_token_id),
        )

    def state_dict(self):
        return self.sampler.state_dict()

    def load_state_dict(self, state):
        if hasattr(self, "sampler"):
            self.sampler.load_state_dict(state)
        else:
            self.pending_state = state
