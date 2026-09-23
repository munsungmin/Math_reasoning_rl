from .base import ModelObservation


class WeightNorm(ModelObservation):
    def __init__(self, every_n_steps=20, trainable_only=True):
        super().__init__(every_n_steps)
        self.trainable_only = trainable_only

    def observe(self, model):
        import torch

        norms = {}
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if not self.trainable_only or parameter.requires_grad:
                    norms[name] = float(
                        torch.linalg.vector_norm(parameter.detach().float())
                    )
        if not norms:
            raise ValueError("WeightNorm selected no parameters")
        return {
            "parameters": norms,
            "total_norm": sum(v * v for v in norms.values()) ** 0.5,
        }
