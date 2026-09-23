"""Read-only views preserve the adaptation parameter selection after inference freezing."""


class InferenceModelView:
    def __init__(self, model):
        self.model = model

    def named_parameters(self):
        has_adapter = hasattr(self.model, "peft_config")
        for name, parameter in self.model.named_parameters():
            selected = "lora_" in name if has_adapter else True
            # A detached view changes no flag, tensor value or gradient of the live model.
            yield name, parameter.detach().requires_grad_(selected)

    def named_modules(self):
        return self.model.named_modules()


def observe_loaded_model(model, callbacks, output, *, stage, step=0):
    from scripts.utils.callbacks import build_observations
    from scripts.utils.logger import ResultWriter

    observations = build_observations(callbacks)
    observations.run(
        InferenceModelView(model),
        step=step,
        stage=stage,
        writer=ResultWriter(output),
        force=True,
    )
