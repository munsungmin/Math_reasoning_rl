"""Initialization is determined by artifact format, never by prior SFT/RL history."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    path: str
    adapter: str | None = None
    tokenizer: str | None = None

    @classmethod
    def resolve(cls, initialization, *, default_base=None):
        source = Path(initialization["path"]).expanduser().resolve()
        adapter = initialization.get("adapter")
        tokenizer = initialization.get("tokenizer")
        if not source.is_dir():
            raise FileNotFoundError(f"Model artifact not found: {source}")
        manifest = source / "model_manifest.json"
        if manifest.exists():
            metadata = json.loads(manifest.read_text())
            if metadata.get("format") == "peft":
                adapter = str(source)
                source = Path(metadata["base_model"])
                tokenizer = tokenizer or metadata.get("tokenizer")
        elif (source / "adapter_config.json").is_file():
            if adapter:
                raise ValueError(
                    "An adapter artifact cannot also specify a second adapter"
                )
            adapter = str(source)
            info = json.loads((source / "adapter_config.json").read_text())
            source = (
                Path(default_base or info["base_model_name_or_path"])
                .expanduser()
                .resolve()
            )
        if not (source / "config.json").is_file():
            raise FileNotFoundError(f"Base model config missing: {source}")
        if adapter:
            adapter_path = Path(adapter).expanduser().resolve()
            info = json.loads((adapter_path / "adapter_config.json").read_text())
            weights = (
                adapter_path / "adapter_model.safetensors",
                adapter_path / "adapter_model.bin",
            )
            if not any(p.is_file() for p in weights):
                raise FileNotFoundError(f"Adapter weights missing: {adapter_path}")
            declared = Path(info.get("base_model_name_or_path", ""))
            if (
                declared.is_absolute()
                and declared.exists()
                and declared.resolve() != source.resolve()
            ):
                raise ValueError("Adapter base model differs from the selected base")
            adapter = str(adapter_path)
            if tokenizer is None and (adapter_path / "tokenizer_config.json").is_file():
                tokenizer = adapter
        return cls(str(source), adapter, str(tokenizer or source))

    def validate_lora(self, config):
        if not self.adapter:
            return
        actual = json.loads((Path(self.adapter) / "adapter_config.json").read_text())
        for key, desired in (("r", config["rank"]), ("lora_alpha", config["alpha"])):
            if actual.get(key) != desired:
                raise ValueError(
                    f"Adapter {key}={actual.get(key)} differs from requested {desired}"
                )
        if set(actual.get("target_modules", [])) != set(config["target_modules"]):
            raise ValueError("Adapter target_modules differ from configuration")

    def validate_native_tokenizer(self):
        if self.tokenizer == self.path:
            return
        from transformers import AutoTokenizer

        base = AutoTokenizer.from_pretrained(self.path, local_files_only=True)
        selected = AutoTokenizer.from_pretrained(self.tokenizer, local_files_only=True)
        if base.get_vocab() != selected.get_vocab() or any(
            getattr(base, key) != getattr(selected, key)
            for key in ("bos_token_id", "eos_token_id")
        ):
            raise ValueError(
                "Native RL requires the same token IDs/vocabulary as the base model"
            )
