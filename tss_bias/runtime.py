"""Bootstrap helpers for legacy analysis scripts."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

from tss_bias.config import REPO_ROOT, ensure_workdir_dirs, load_config


def bootstrap(config_path: Optional[str] = None, chdir_work: bool = True) -> dict[str, Any]:
    """Load config, ensure dirs, optionally chdir to work_root, fix sys.path."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    cfg = load_config(config_path or os.environ.get("TSS_CONFIG"))
    ensure_workdir_dirs(cfg)
    if chdir_work:
        work = Path(cfg["paths"]["work_root"])
        work.mkdir(parents=True, exist_ok=True)
        os.chdir(work)
        # Symlink in-repo assets expected at relative paths by legacy scripts
        _ensure_link(work / "data", REPO_ROOT / "data")
        _ensure_link(work / "sample_lists", REPO_ROOT / "data" / "sample_lists")
        # Legacy scripts still open sample_lists/ and src/
        _ensure_link(work / "src", REPO_ROOT / "data" / "reference")
    return cfg


def _ensure_link(link: Path, target: Path) -> None:
    if link.exists() or link.is_symlink():
        return
    try:
        link.symlink_to(target)
    except OSError:
        pass


def mpa_path(cfg: dict[str, Any]) -> str:
    return str(cfg["paths"]["mpa_merged"])
