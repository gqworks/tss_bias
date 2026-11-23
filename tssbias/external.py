"""Subprocess helpers. Commands are logged and failures keep the exit code."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from tssbias.errors import ToolError
from tssbias.logutil import get_logger

LOG = get_logger("tssbias.external")


def require_tool(name: str) -> str:
    """Return the executable path or raise ToolError.

    Inputs:
        name: program name on PATH (``bowtie2``, ``prodigal``, ...).
    Outputs:
        Absolute path from ``shutil.which``.
    """
    found = shutil.which(name)
    if not found:
        raise ToolError(
            f"Required program {name!r} is not on PATH. "
            "Install the pinned conda environment (environment.yml) on Gadi "
            "and rerun. No results were written."
        )
    return found


def run_command(args: list[str], *, cwd: Path | None = None) -> None:
    """Run one program. Raise ToolError on a non-zero exit.

    Inputs:
        args: argv, without a shell.
        cwd: optional working directory.
    Outputs:
        None. stdout/stderr are inherited so the scheduler log keeps them.
    """
    LOG.info("command: %s", " ".join(args))
    completed = subprocess.run(args, cwd=cwd, check=False)
    if completed.returncode != 0:
        raise ToolError(f"{args[0]} exited {completed.returncode}")


def run_pipeline(stages: list[list[str]]) -> None:
    """Run programs joined by pipes. Raise ToolError if any stage fails.

    Inputs:
        stages: argv lists. stdout of each stage feeds stdin of the next.
    Outputs:
        None. The last stage inherits this process stdout.
    """
    LOG.info("pipeline: %s", " | ".join(" ".join(stage) for stage in stages))
    processes: list[subprocess.Popen[bytes]] = []
    previous: subprocess.Popen[bytes] | None = None
    try:
        for index, stage in enumerate(stages):
            last = index == len(stages) - 1
            proc = subprocess.Popen(
                stage,
                stdin=previous.stdout if previous is not None else None,
                stdout=None if last else subprocess.PIPE,
            )
            if previous is not None and previous.stdout is not None:
                previous.stdout.close()
            processes.append(proc)
            previous = proc
        return_codes = [proc.wait() for proc in processes]
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.kill()
    failed = [code for code in return_codes if code != 0]
    if failed:
        raise ToolError(
            "pipeline failed with exit codes "
            + ", ".join(f"{stage[0]}={code}" for stage, code in zip(stages, return_codes))
        )
