"""Research callbacks inspect model state; they do not control the training objective."""


class ModelObservation:
    def __init__(self, every_n_steps=20):
        self.every_n_steps = int(every_n_steps)
        if self.every_n_steps <= 0:
            raise ValueError("Observation interval must be positive")

    def due(self, step):
        return step % self.every_n_steps == 0

    def observe(self, model):
        raise NotImplementedError


class Observations:
    def __init__(self, callbacks):
        self.callbacks = dict(callbacks)
        self.completed = set()

    def run(self, model, *, step, stage, writer, force=False):
        from scripts.utils.seed import isolated_rng

        for name, callback in self.callbacks.items():
            key = (name, stage, int(step))
            if key in self.completed or not (force or callback.due(step)):
                continue
            with isolated_rng():
                values = callback.observe(model)
            writer.log({name: values}, step=step, stage=stage, source="model")
            self.completed.add(key)

    def state_dict(self):
        return {"completed": sorted(self.completed)}

    def load_state_dict(self, state):
        self.completed = {tuple(key) for key in state["completed"]}
