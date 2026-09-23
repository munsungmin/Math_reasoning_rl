"""Own a worker process group and bound cleanup, including interrupted runs."""

import os
import signal
import subprocess


def stop_group(process, timeout=5):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        # A process blocked in kernel/driver I/O can remain pending after SIGKILL.
        print(
            f"Worker group {process.pid}: termination requested; kernel I/O cleanup is pending",
            flush=True,
        )


def run_worker_command(command, *, cwd, env):
    process = subprocess.Popen(command, cwd=cwd, env=env, start_new_session=True)
    try:
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
    finally:
        stop_group(process)
