"""Load and resolve YAML configuration for portable (Gadi) runs."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Union

import yaml

PathLike = Union[str, Path]

# Repo root = parent of the tss_bias package directory
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config" / "default.yaml"


def _deep_update(base: MutableMapping[str, Any], overlay: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            _deep_update(base[key], value)  # type: ignore[arg-type]
        else:
            base[key] = value
    return base


_BASH_DEFAULT = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env_defaults(value: str) -> str:
    """Expand ${VAR} and ${VAR:-default} (bash-style); leave unknown bare $VAR to expandvars."""

    def repl(match: re.Match[str]) -> str:
        var, default = match.group(1), match.group(2)
        if var in os.environ and os.environ[var] != "":
            return os.environ[var]
        if default is not None:
            return default
        return match.group(0)

    return _BASH_DEFAULT.sub(repl, value)


def _expand(value: str, env: Mapping[str, str]) -> str:
    """Expand ${VAR:-default} / $VAR and {placeholders} from env-like map."""
    expanded = _expand_env_defaults(value)
    expanded = os.path.expandvars(expanded)
    try:
        expanded = expanded.format_map(env)
    except (KeyError, ValueError):
        pass
    return expanded


def _resolve_tree(node: Any, env: Mapping[str, str]) -> Any:
    if isinstance(node, str):
        return _expand(node, env)
    if isinstance(node, list):
        return [_resolve_tree(x, env) for x in node]
    if isinstance(node, dict):
        return {k: _resolve_tree(v, env) for k, v in node.items()}
    return node


def load_config(
    path: Optional[PathLike] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Load default.yaml (+ optional overlay), expand env/placeholders.

    Environment variables:
      TSS_CONFIG          path to overlay YAML
      TSS_DATA_ROOT       overrides paths.data_root
      TSS_SCRATCH_ROOT    overrides paths.scratch_root
      TSS_PROJECT         NCI project code (e.g. ab12)
    """
    cfg_path = Path(path or os.environ.get("TSS_CONFIG", DEFAULT_CONFIG))
    with open(cfg_path) as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh) or {}

    overlay_env = os.environ.get("TSS_CONFIG_OVERLAY")
    if overlay_env:
        with open(overlay_env) as fh:
            _deep_update(cfg, yaml.safe_load(fh) or {})

    if overrides:
        _deep_update(cfg, overrides)

    paths = cfg.setdefault("paths", {})
    if os.environ.get("TSS_DATA_ROOT"):
        paths["data_root"] = os.environ["TSS_DATA_ROOT"]
    if os.environ.get("TSS_SCRATCH_ROOT"):
        paths["scratch_root"] = os.environ["TSS_SCRATCH_ROOT"]
    if os.environ.get("TSS_PROJECT"):
        cfg.setdefault("gadi", {})["project"] = os.environ["TSS_PROJECT"]

    # Build format map from path keys + gadi.project
    fmt: dict[str, str] = {}
    for k, v in paths.items():
        if isinstance(v, str):
            fmt[k] = os.path.expandvars(v)
    gadi = cfg.get("gadi") or {}
    if "project" in gadi:
        fmt["project"] = str(gadi["project"])
    fmt.setdefault("repo_root", str(REPO_ROOT))

    # Two-pass expand so nested {data_root}/foo works
    paths_resolved = {k: _expand(str(v), fmt) if isinstance(v, str) else v for k, v in paths.items()}
    fmt.update({k: str(v) for k, v in paths_resolved.items()})
    cfg["paths"] = paths_resolved
    cfg = _resolve_tree(cfg, fmt)
    cfg["_repo_root"] = str(REPO_ROOT)
    cfg["_config_path"] = str(cfg_path.resolve())
    return cfg


def ensure_workdir_dirs(cfg: Mapping[str, Any]) -> None:
    """Create standard output directories under work_root / scratch as needed."""
    paths = cfg["paths"]
    for key in ("work_root", "refseq_dir", "matrices_dir", "beds_dir", "plots_dir", "mpa_dir", "kallisto_dir"):
        p = paths.get(key)
        if p:
            Path(p).mkdir(parents=True, exist_ok=True)
    for key in ("bams_dir", "fastq_scratch"):
        p = paths.get(key)
        if p:
            Path(p).mkdir(parents=True, exist_ok=True)
