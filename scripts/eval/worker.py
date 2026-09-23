import sys

from omegaconf import OmegaConf

from scripts.utils.manifest import record_environment


def main():
    from scripts.utils.config import register_resolvers

    register_resolvers()
    config = OmegaConf.load(sys.argv[1])
    record_environment(config.runtime.output_dir)
    from .pipeline import run

    run(config)


if __name__ == "__main__":
    main()
