"""Generate once; graders and metrics consume the resulting records independently."""

from scripts.utils.seed import isolated_rng


def generate(model, tokenizer, rows, sampling):
    import torch

    sampling.validate()
    requests = [(row, i) for i in range(sampling.n) for row in rows]
    was_training = model.training
    old_cache = model.config.use_cache
    old_padding = tokenizer.padding_side
    try:
        model.eval()
        model.config.use_cache = True
        tokenizer.padding_side = "left"
        with isolated_rng(sampling.seed), torch.inference_mode():
            for start in range(0, len(requests), sampling.batch_size):
                batch = requests[start : start + sampling.batch_size]
                inputs = tokenizer(
                    [row["prompt"] for row, _ in batch],
                    padding=True,
                    return_tensors="pt",
                    add_special_tokens=False,
                    return_token_type_ids=False,
                ).to(model.device)
                output = model.generate(
                    **inputs,
                    **sampling.transformers_kwargs(),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )[:, inputs.input_ids.shape[1] :]
                for (row, sample_index), tokens in zip(batch, output, strict=True):
                    tokens = tokens.tolist()
                    ended = tokenizer.eos_token_id in tokens
                    length = (
                        tokens.index(tokenizer.eos_token_id) + 1
                        if ended
                        else len(tokens)
                    )
                    result = {
                        "sample_id": row["sample_id"],
                        "sample_index": sample_index,
                        "dataset": row.get("dataset", "math"),
                        "split": row.get("split", "test"),
                        "input": row["prompt"],
                        "output": tokenizer.decode(
                            tokens[:length], skip_special_tokens=True
                        ),
                        "generated_tokens": length,
                        "hit_length_limit": not ended,
                        "seed": sampling.seed,
                        "temperature": sampling.temperature,
                    }
                    if "answer" in row:
                        result["answer"] = row["answer"]
                    yield result
    finally:
        model.train(was_training)
        model.config.use_cache = old_cache
        tokenizer.padding_side = old_padding
