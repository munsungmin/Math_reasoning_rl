"""Read the pinned native FSDP LoRA checkpoint without rounding model weights.

Native training still owns checkpointing/resume. This adapter exposes completed
one-dimensional FSDP saves to independent model observation and inference.
"""

import json
from pathlib import Path
from types import SimpleNamespace


def prepare_initial_adapter(source, output, base_model):
    """Take SFT/RL adapter weights as initialization with this RL job's dropout=0.

    The source artifact stays immutable. The pinned engine loads dropout from the
    PEFT file, so changing only the project YAML would silently retain SFT dropout.
    """
    import shutil

    from scripts.utils.io import sha256, write_json

    source, output = Path(source), Path(output)
    metadata = json.loads((source / "adapter_config.json").read_text())
    unsupported = [
        k
        for k in (
            "use_dora",
            "use_rslora",
            "modules_to_save",
            "rank_pattern",
            "alpha_pattern",
        )
        if metadata.get(k)
    ]
    if unsupported or metadata.get("bias", "none") != "none":
        raise ValueError(
            f"Native adapter observation/export requires ordinary LoRA: {unsupported}"
        )
    output.mkdir(parents=True, exist_ok=False)
    original_dropout = metadata.get("lora_dropout", 0.0)
    metadata.update(lora_dropout=0.0, base_model_name_or_path=str(base_model))
    write_json(output / "adapter_config.json", metadata)
    weights = {}
    for path in source.iterdir():
        if path.is_file() and path.name.startswith("adapter_model."):
            shutil.copy2(path, output / path.name)
            weights[path.name] = sha256(path)
    return {
        "source": str(source),
        "path": str(output),
        "source_dropout": original_dropout,
        "training_dropout": 0.0,
        "weight_sha256": weights,
    }


def load_adapter_state(actor):
    import torch
    from torch.distributed.tensor import DTensor

    actor = Path(actor)
    world = json.loads((actor / "fsdp_config.json").read_text())["world_size"]
    metadata = json.loads((actor / "lora_train_meta.json").read_text())
    shards, specifications = {}, {}
    for rank in range(world):
        state = torch.load(
            actor / f"model_world_size_{world}_rank_{rank}.pt",
            map_location="cpu",
            weights_only=False,
        )
        selected = {
            k: v for k, v in state.items() if ".lora_A." in k or ".lora_B." in k
        }
        if not selected or (rank and selected.keys() != shards.keys()):
            raise ValueError("Inconsistent/missing LoRA keys in native checkpoint")
        for key, value in selected.items():
            placement = None
            if isinstance(value, DTensor):
                if value.device_mesh.ndim != 1 or len(value.placements) != 1:
                    raise ValueError(
                        "Adapter export currently supports a one-dimensional FSDP mesh"
                    )
                placement = value.placements[0]
                if not (placement.is_shard() or placement.is_replicate()):
                    raise ValueError(f"Unsupported placement for {key}: {placement}")
                spec = (tuple(value.shape), value.dtype, str(placement))
                local = value.to_local().detach().clone()
            else:
                spec = (tuple(value.shape), value.dtype, None)
                local = value.detach().clone()
            if rank and spec != specifications[key][0]:
                raise ValueError(f"Inconsistent shard metadata: {key}")
            specifications[key] = (spec, placement)
            shards.setdefault(key, []).append(local)
        del state, selected
    result = {}
    for key, tensors in shards.items():
        (shape, _, _), placement = specifications[key]
        if placement is not None and placement.is_shard():
            merged = torch.cat(tensors, dim=placement.dim)
            merged = merged.narrow(placement.dim, 0, shape[placement.dim])
        else:
            if any(not torch.equal(tensors[0], t) for t in tensors[1:]):
                raise ValueError(f"Replicated adapter tensors disagree: {key}")
            merged = tensors[0]
        if tuple(merged.shape) != shape:
            raise ValueError(f"Incorrect reconstructed shape: {key}")
        result[key] = merged.contiguous()
    return result, metadata


class AdapterModelView:
    def __init__(self, state, metadata):
        self.state, self.metadata = state, metadata

    def named_parameters(self):
        for name, value in self.state.items():
            yield name, value.detach().requires_grad_(True)

    def named_modules(self):
        for name, a in self.state.items():
            if ".lora_A.default.weight" not in name:
                continue
            prefix = name.removesuffix(".lora_A.default.weight")
            b = self.state[prefix + ".lora_B.default.weight"]
            rank = self.metadata["r"]
            if a.shape[0] != rank or b.shape[1] != rank:
                raise ValueError("Adapter rank differs from saved metadata")
            yield (
                prefix,
                SimpleNamespace(
                    lora_A={"default": SimpleNamespace(weight=a)},
                    lora_B={"default": SimpleNamespace(weight=b)},
                    scaling={"default": self.metadata["lora_alpha"] / rank},
                ),
            )


def export_adapter(actor, output, base_model, *, state=None, metadata=None):
    from peft import LoraConfig
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    from scripts.utils.io import write_json

    actor, output = Path(actor), Path(output)
    if state is None:
        state, metadata = load_adapter_state(actor)
    targets = sorted({k.split(".lora_")[0].rsplit(".", 1)[-1] for k in state})
    output.mkdir(parents=True, exist_ok=False)
    adapter = LoraConfig(
        r=metadata["r"],
        lora_alpha=metadata["lora_alpha"],
        target_modules=targets,
        lora_dropout=0.0,
        task_type="CAUSAL_LM",
        base_model_name_or_path=str(base_model),
    )
    adapter.save_pretrained(output)
    save_file(
        {k.replace(".default.weight", ".weight"): v for k, v in state.items()},
        output / "adapter_model.safetensors",
    )
    AutoTokenizer.from_pretrained(
        actor / "huggingface", local_files_only=True
    ).save_pretrained(output)
    write_json(
        output / "model_manifest.json",
        {
            "version": 1,
            "format": "peft",
            "base_model": str(base_model),
            "tokenizer": str(output.resolve()),
            "source_checkpoint": str(actor.resolve()),
            "weight_dtypes": sorted({str(v.dtype) for v in state.values()}),
        },
    )
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export a completed native RL actor as a reusable PEFT artifact"
    )
    parser.add_argument("actor", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--base-model", required=True)
    arguments = parser.parse_args()
    export_adapter(arguments.actor, arguments.output, arguments.base_model)
