"""Batch order belongs to data, with a cursor advanced by consumed batches only."""

import math
import random


class ResumableBatchSampler:
    def __init__(self, size, batch_size, seed, rank=0, world_size=1):
        self.size, self.batch_size, self.seed = int(size), int(batch_size), int(seed)
        self.rank, self.world_size = rank, world_size
        self.epoch, self.cursor = 0, 0
        if min(self.size, self.batch_size, self.world_size) < 1:
            raise ValueError("Sampler size/batch/world size must be positive")

    def order(self):
        indices = list(range(self.size))
        random.Random(self.seed + self.epoch).shuffle(indices)
        total = math.ceil(self.size / self.world_size) * self.world_size
        indices = (indices * math.ceil(total / self.size))[:total]
        return indices[self.rank : total : self.world_size]

    def __iter__(self):
        indices = self.order()
        for start in range(self.cursor, len(indices), self.batch_size):
            yield indices[start : start + self.batch_size]

    def __len__(self):
        return math.ceil(math.ceil(self.size / self.world_size) / self.batch_size)

    def acknowledge(self, count):
        self.cursor += int(count)
        if self.cursor > len(self.order()):
            raise ValueError("Sampler consumed more records than its epoch contains")

    def set_epoch(self, epoch):
        if self.epoch != epoch:
            self.epoch, self.cursor = int(epoch), 0

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "cursor": self.cursor,
            "seed": self.seed,
            "size": self.size,
            "world_size": self.world_size,
            "batch_size": self.batch_size,
        }

    def load_state_dict(self, state):
        for key in ("seed", "size", "world_size", "batch_size"):
            if state[key] != getattr(self, key):
                raise ValueError(f"Cannot resume sampler with changed {key}")
        self.epoch, self.cursor = state["epoch"], state["cursor"]
