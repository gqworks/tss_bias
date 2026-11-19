"""Path helpers that keep analysis code free of cluster-absolute roots."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def p(cfg: Mapping[str, Any], key: str) -> Path:
    return Path(cfg["paths"][key])


def matrix_path(cfg: Mapping[str, Any], asm: str, sample: str) -> Path:
    return p(cfg, "matrices_dir") / f"{asm}_{sample}.csv"


def bam_path(cfg: Mapping[str, Any], asm: str, sample: str) -> Path:
    return p(cfg, "bams_dir") / f"{asm}_{sample}.sorted.bam"


def bed_bw_path(cfg: Mapping[str, Any], asm: str, sample: str) -> Path:
    return p(cfg, "beds_dir") / f"{asm}_{sample}.bw"


def genome_dir(cfg: Mapping[str, Any], asm: str) -> Path:
    return p(cfg, "refseq_dir") / asm


def sample_list_path(cfg: Mapping[str, Any], name: str) -> Path:
    """Resolve a sample list by basename under configured sample_lists_dir."""
    base = p(cfg, "sample_lists_dir")
    candidate = base / name
    if candidate.exists():
        return candidate
    # Allow legacy repo-relative sample_lists/
    legacy = Path(cfg["_repo_root"]) / "sample_lists" / name
    if legacy.exists():
        return legacy
    raise FileNotFoundError(f"Sample list not found: {name} (looked in {base} and legacy sample_lists/)")


def reference_path(cfg: Mapping[str, Any], name: str) -> Path:
    return p(cfg, "reference_dir") / name


def mpa_merged_path(cfg: Mapping[str, Any]) -> Path:
    return Path(cfg["paths"]["mpa_merged"])
