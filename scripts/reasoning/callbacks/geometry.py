from ..metric.functional.geometry import lora_singular_values, spectrum_statistics
from .base import ModelObservation


class AdapterGeometry(ModelObservation):
    def observe(self, model):
        result = {}
        # PEFT modules carry the effective scaling, including rslora/custom patterns.
        for module_name, module in model.named_modules():
            if not hasattr(module, "lora_A"):
                continue
            for adapter_name, a in module.lora_A.items():
                b = module.lora_B[adapter_name]
                singular = lora_singular_values(
                    a.weight, b.weight, module.scaling[adapter_name]
                )
                result[f"{module_name}/{adapter_name}"] = spectrum_statistics(singular)
        if not result:
            raise ValueError("AdapterGeometry requires a PEFT LoRA model")
        return result
