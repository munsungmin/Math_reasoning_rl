"""Model construction is called explicitly inside the selected worker environment."""

from .spec import ModelSpec


def load_model(config, *, trainable, device="cpu"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = ModelSpec.resolve(config["initialization"])
    dtype = getattr(torch, config.get("dtype", "float32"))
    tokenizer = AutoTokenizer.from_pretrained(spec.tokenizer, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        spec.path,
        torch_dtype=dtype,
        attn_implementation=config.get("attention", "sdpa"),
        local_files_only=True,
    )
    lora = config.get("adapter", {})
    if spec.adapter:
        from peft import PeftModel

        spec.validate_lora(lora)
        model = PeftModel.from_pretrained(model, spec.adapter, is_trainable=trainable)
    elif trainable and lora.get("rank", 0) > 0:
        from scripts.reasoning.modules.adapters import attach_lora

        model = attach_lora(model, lora)
    if trainable and config.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
    model.config.use_cache = not trainable
    model.to(device)
    model.train(trainable)
    if not trainable:
        model.requires_grad_(False)
    return model, tokenizer, spec


def export_model(model, tokenizer, spec, output):
    from pathlib import Path

    from scripts.utils.io import write_json

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    write_json(
        output / "model_manifest.json",
        {
            "version": 1,
            "format": "peft"
            if (output / "adapter_config.json").exists()
            else "huggingface",
            "base_model": spec.path,
            "tokenizer": str(output.resolve()),
        },
    )
    return output
