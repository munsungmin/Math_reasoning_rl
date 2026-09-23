"""Trainable adaptation components are constructed independently of the LM loader."""


def attach_lora(model, config):
    from peft import LoraConfig, get_peft_model

    return get_peft_model(
        model,
        LoraConfig(
            r=config["rank"],
            lora_alpha=config["alpha"],
            lora_dropout=config.get("dropout", 0.0),
            target_modules=list(config["target_modules"]),
            task_type="CAUSAL_LM",
        ),
    )
