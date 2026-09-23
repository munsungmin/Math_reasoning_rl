import json
import subprocess

from scripts.utils.paths import REPO


class SubprocessGrader:
    def __init__(self, python, script=None, timeout=90):
        self.python = str(python)
        self.script = str(script or REPO / "scripts/eval/graders/worker.py")
        self.timeout = timeout

    def __call__(self, rows):
        rows = list(rows)
        if not rows:
            return []
        result = subprocess.run(
            [self.python, self.script],
            input=json.dumps(rows),
            text=True,
            capture_output=True,
            timeout=self.timeout,
            check=False,
        )
        if result.returncode:
            raise RuntimeError("MATH grader failed: " + result.stderr[-4000:])
        results = json.loads(result.stdout)
        if len(results) != len(rows):
            raise ValueError("Grader result count differs from its input")
        return results
