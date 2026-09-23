"""Observe completed native checkpoints; never reach into live training workers."""

import json
from pathlib import Path

from scripts.utils.callbacks import build_observations
from scripts.utils.io import write_json
from scripts.utils.logger import ResultWriter

from .adapter import AdapterModelView, export_adapter, load_adapter_state


def observe_checkpoint(run, cfg, step, *, final=False):
    project = cfg.math_reasoning
    run = Path(run)
    output = run / "observations"
    observations = build_observations(project.callbacks)
    state_path = output / "state.json"
    if state_path.exists():
        observations.load_state_dict(json.loads(state_path.read_text()))
    due = any(c.due(step) or final for c in observations.callbacks.values())
    target = run / "exports" / f"step_{step}"
    if not due and not final:
        return
    actor = Path(cfg.trainer.default_local_dir) / f"global_step_{step}" / "actor"
    state, metadata = load_adapter_state(actor)
    observations.run(
        AdapterModelView(state, metadata),
        step=step,
        stage="train",
        writer=ResultWriter(output),
        force=final,
    )
    write_json(state_path, observations.state_dict())
    if final and not target.exists():
        export_adapter(
            actor,
            target,
            cfg.actor_rollout_ref.model.path,
            state=state,
            metadata=metadata,
        )
        write_json(
            run / "checkpoint_export.json",
            {
                "global_step": step,
                "export": str(target),
                "checkpoint": str(actor.parent),
            },
        )
