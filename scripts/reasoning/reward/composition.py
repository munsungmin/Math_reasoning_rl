class WeightedReward:
    def __init__(self, components, weights):
        self.components, self.weights = dict(components), dict(weights)
        if self.components.keys() != self.weights.keys():
            raise ValueError("Each reward component requires an explicit weight")

    def __call__(self, completion, judgement):
        parts = {
            name: float(fn(completion, judgement))
            for name, fn in self.components.items()
        }
        return {
            "score": sum(parts[name] * self.weights[name] for name in parts),
            "acc": float(judgement["correct"]),
            **parts,
        }
