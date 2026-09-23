import lightning.pytorch as pl


class ObservationBridge(pl.Callback):
    def __init__(self, observations, writer):
        self.observations, self.writer = observations, writer

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Consume progress before ModelCheckpoint's batch-end save.
        trainer.datamodule.sampler.acknowledge(len(batch["sample_id"]))
        if trainer.is_global_zero and trainer.global_step > 0:
            self.observations.run(
                pl_module.model,
                step=trainer.global_step,
                stage="train",
                writer=self.writer,
            )

    def on_fit_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            self.observations.run(
                pl_module.model,
                step=trainer.global_step,
                stage="train",
                writer=self.writer,
                force=True,
            )

    def state_dict(self):
        return self.observations.state_dict()

    def load_state_dict(self, state):
        self.observations.load_state_dict(state)
