"""Errors for missing data and external programs.

These are the only failures the commands should raise when a Gadi path has
not been staged. They do not invent placeholder results.
"""

from __future__ import annotations

from pathlib import Path


class MissingInputError(FileNotFoundError):
    """Raised when a command cannot run because an input file is absent.

    Inputs:
        what: one-line description of the analysis that stopped.
        paths: files or directories that were required and missing.
    """

    def __init__(self, what: str, paths: list[Path | str]):
        lines = [what, "Missing inputs:"]
        for path in paths:
            lines.append(f"  - {path}")
        lines.append(
            "Stage these paths in the config (TSS_CONFIG, TSS_DATA_ROOT, "
            "TSS_SCRATCH_ROOT) and rerun. No results were written."
        )
        super().__init__("\n".join(lines))
        self.paths = [Path(p) for p in paths]


class ToolError(RuntimeError):
    """An external program is missing or returned a non-zero status."""


def require_paths(what: str, paths: list[Path]) -> None:
    """Raise MissingInputError if any path is absent.

    Inputs:
        what: analysis name for the error message.
        paths: files that must already exist.
    Outputs:
        None. Raises MissingInputError listing every missing path.
    """
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise MissingInputError(what, missing)
