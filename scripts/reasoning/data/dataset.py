"""Supervised encoding and padding; only target tokens contribute to the loss."""


class SupervisedRows:
    def __init__(self, rows, tokenizer, max_tokens):
        self.rows, self.tokenizer, self.max_tokens = (
            list(rows),
            tokenizer,
            int(max_tokens),
        )
        if tokenizer.eos_token_id is None:
            raise ValueError("A supervised tokenizer requires EOS")
        for row in self.rows:
            if not row.get("target"):
                raise ValueError(f"Missing supervised target: {row['sample_id']}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        prompt = self.tokenizer.encode(row["prompt"], add_special_tokens=False)
        target = self.tokenizer.encode(row["target"], add_special_tokens=False) + [
            self.tokenizer.eos_token_id
        ]
        if not prompt or len(prompt) + len(target) > self.max_tokens:
            raise ValueError(f"Invalid/overlong supervised sample {row['sample_id']}")
        return {
            "input_ids": prompt + target,
            "labels": [-100] * len(prompt) + target,
            "prompt_length": len(prompt),
            "sample_id": row["sample_id"],
            "label": row.get("label", -1),
        }


class SupervisedCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, rows):
        import torch

        size = max(len(r["input_ids"]) for r in rows)
        ids, labels, masks = [], [], []
        for row in rows:
            padding = size - len(row["input_ids"])
            ids.append(row["input_ids"] + [self.pad_token_id] * padding)
            labels.append(row["labels"] + [-100] * padding)
            masks.append([1] * len(row["input_ids"]) + [0] * padding)
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(masks),
            "sample_id": [r["sample_id"] for r in rows],
            "prompt_length": torch.tensor([r["prompt_length"] for r in rows]),
            "label": torch.tensor([r["label"] for r in rows]),
        }
