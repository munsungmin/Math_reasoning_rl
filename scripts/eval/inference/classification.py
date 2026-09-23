"""Code-token classification uses the same label contract as supervised training."""


def label_token_ids(tokenizer, labels):
    ids = [tokenizer.encode(label, add_special_tokens=False) for label in labels]
    if any(len(token_ids) != 1 for token_ids in ids) or len(
        {tuple(x) for x in ids}
    ) != len(ids):
        raise ValueError("Classification labels must map to distinct single tokens")
    return [x[0] for x in ids]


def classify(model, tokenizer, rows, *, labels, batch_size=8):
    import torch

    code_ids = label_token_ids(tokenizer, labels)
    old_mode, old_padding = model.training, tokenizer.padding_side
    try:
        model.eval()
        tokenizer.padding_side = "left"
        with torch.inference_mode():
            for start in range(0, len(rows), batch_size):
                batch = rows[start : start + batch_size]
                inputs = tokenizer(
                    [r["prompt"] for r in batch],
                    padding=True,
                    return_tensors="pt",
                    add_special_tokens=False,
                    return_token_type_ids=False,
                ).to(model.device)
                logits = model(**inputs).logits[:, -1, code_ids].float()
                for row, scores in zip(batch, logits, strict=True):
                    result = {
                        "sample_id": row["sample_id"],
                        "sample_index": 0,
                        "dataset": row.get("dataset", "classification"),
                        "split": row.get("split", "test"),
                        "prediction": int(scores.argmax()),
                        "scores": scores.cpu().tolist(),
                        "log_probs": scores.log_softmax(-1).cpu().tolist(),
                    }
                    if "label" in row:
                        result["label"] = row["label"]
                    yield result
    finally:
        model.train(old_mode)
        tokenizer.padding_side = old_padding
