"""Construct a Lightning SFT job from independently selected components."""

from pathlib import Path


def run(config):
    import lightning.pytorch as pl
    from hydra.utils import instantiate
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger, WandbLogger

    from scripts.reasoning.model.loading import export_model, load_model
    from scripts.utils.callbacks import build_observations
    from scripts.utils.io import write_json
    from scripts.utils.logger import ResultWriter

    from .data import LightningData
    from .observation import ObservationBridge

    output = Path(config.runtime.output_dir)
    pl.seed_everything(config.model_seed, workers=True)
    model, tokenizer, spec = load_model(config.model, trainable=True)
    data = instantiate(config.data)
    module = LightningData(
        data,
        tokenizer,
        config.train_config,
        config.data_seed,
        config.runtime.num_workers,
    )
    task = instantiate(
        config.task,
        model=model,
        tokenizer=tokenizer,
        loss=instantiate(config.loss),
        optimizer=instantiate(config.optimizer),
        scheduler=config.lr_scheduler,
        train=config.train_config,
        sampling=config.test_config,
        grader=config.grader,
    )
    observer = ObservationBridge(
        build_observations(config.callbacks), ResultWriter(output / "observations")
    )
    checkpoint = ModelCheckpoint(
        dirpath=str(output / "checkpoints"),
        filename="step-{step}",
        every_n_train_steps=config.trainer.save_every,
        save_top_k=-1,
        save_last=True,
    )
    loggers = [CSVLogger(str(output), name="logs")]
    if config.logger.mode != "disabled":
        loggers.append(
            WandbLogger(
                project=config.logger.project,
                entity=config.logger.entity,
                name=output.name,
                id=output.name,
                save_dir=str(output),
                offline=config.logger.mode == "offline",
                resume="allow",
            )
        )
    devices = len(config.runtime.gpu_ids) if config.trainer.accelerator == "gpu" else 1
    trainer = pl.Trainer(
        accelerator=config.trainer.accelerator,
        devices=devices,
        strategy="ddp" if devices > 1 else "auto",
        precision=config.trainer.precision,
        max_steps=config.train_config.max_steps,
        max_epochs=config.train_config.max_epochs,
        accumulate_grad_batches=config.train_config.accumulate_grad_batches,
        gradient_clip_val=config.train_config.grad_clip,
        check_val_every_n_epoch=config.trainer.validate_every,
        callbacks=[observer, checkpoint],
        logger=loggers,
        default_root_dir=str(output),
        use_distributed_sampler=False,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        enable_progress_bar=False,
    )
    try:
        trainer.fit(task, datamodule=module, ckpt_path=config.resume)
        trainer.save_checkpoint(str(output / "checkpoints/last.ckpt"))
        if trainer.is_global_zero:
            target = output / "exports" / f"step_{trainer.global_step}"
            export_model(task.model, tokenizer, spec, target)
            write_json(
                output / "result.json",
                {
                    "completed": True,
                    "kind": config.task.kind,
                    "global_step": trainer.global_step,
                    "export": str(target),
                    "checkpoint": str(output / "checkpoints/last.ckpt"),
                    "consumed_ids": task.consumed_ids,
                },
            )
    finally:
        if config.logger.mode != "disabled":
            import wandb

            wandb.finish()
