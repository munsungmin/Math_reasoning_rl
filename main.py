"""Composable experiments, with compatibility for established native verl commands."""

import sys


def main():
    arguments = sys.argv[1:]
    legacy_commands = {"start", "resume", "check", "logs", "status", "test", "help"}
    legacy = bool(arguments and arguments[0] in legacy_commands)
    legacy |= any(a.startswith(("launcher.", "actor_rollout_ref.")) for a in arguments)
    legacy |= any(a.startswith("--config-name=verl_") for a in arguments)
    legacy |= any(
        a in ("--config-name", "-cn")
        and i + 1 < len(arguments)
        and arguments[i + 1].startswith("verl_")
        for i, a in enumerate(arguments)
    )
    if legacy:
        from scripts.train.rl.runner import main as entry
    else:
        from scripts.train.entry import main as entry
    entry()


if __name__ == "__main__":
    main()
