class RLTask:
    """Select the components; the native backend owns the distributed training loop."""

    def __init__(self, kind="rl", val_metrics=None, test_metrics=None):
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics

    def run(self, config):
        import os
        import sys

        if os.path.realpath(sys.executable) != os.path.realpath(config.runtime.python):
            import subprocess
            import tempfile
            from pathlib import Path

            from omegaconf import OmegaConf

            from scripts.utils.paths import REPO

            with tempfile.TemporaryDirectory(prefix="reasoning-rl-config-") as folder:
                path = Path(folder) / "experiment.yaml"
                OmegaConf.save(config, path, resolve=True)
                return subprocess.run(
                    [config.runtime.python, "-m", "scripts.train.rl.task", str(path)],
                    cwd=REPO,
                    check=True,
                )
        from .config import to_native
        from .runner import run_command

        native = to_native(config)
        return run_command(
            "check" if config.command == "check" else "start", settings=native
        )


if __name__ == "__main__":
    import sys

    from omegaconf import OmegaConf

    from scripts.utils.config import register_resolvers

    register_resolvers()
    RLTask().run(OmegaConf.load(sys.argv[1]))
