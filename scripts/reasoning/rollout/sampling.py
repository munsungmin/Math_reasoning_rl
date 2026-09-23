from dataclasses import dataclass


@dataclass
class NativeValidationSampling:
    """Native in-training validation shares the rollout engine's seed/length budget."""

    n: int = 1
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1

    def native(self):
        return Sampling(
            n=self.n, temperature=self.temperature, top_p=self.top_p, top_k=self.top_k
        ).native()


@dataclass
class Sampling:
    n: int = 1
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    max_new_tokens: int = 1024
    batch_size: int = 8
    seed: int = 17

    def validate(self):
        if (
            min(self.n, self.max_new_tokens, self.batch_size) < 1
            or self.temperature < 0
        ):
            raise ValueError(
                "Invalid generation count, length, batch size or temperature"
            )
        if not 0 < self.top_p <= 1 or self.top_k < -1:
            raise ValueError("Invalid top_p/top_k")

    def transformers_kwargs(self):
        self.validate()
        values = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            values.update(
                temperature=self.temperature, top_p=self.top_p, top_k=max(0, self.top_k)
            )
        return values

    def native(self):
        self.validate()
        return {
            "n": self.n,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "do_sample": self.temperature > 0,
        }
