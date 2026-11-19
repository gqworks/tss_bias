"""Pinned RefSeq accessions used across analyses (post–fixed-genomes commit)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from tss_bias.config import REPO_ROOT

DEFAULT_SPECIES_YAML = REPO_ROOT / "config" / "species_accessions.yaml"

NCBI_FTP_PREFIX = "https://ftp.ncbi.nlm.nih.gov/genomes/all"


def accession_to_ftp(asm: str) -> str:
    """Build NCBI FTP directory URL from assembly accession folder name.

    Example: GCF_000012825.1_ASM1282v1
      -> .../GCF/000/012/825/GCF_000012825.1_ASM1282v1
    """
    # asm like GCF_000012825.1_ASM1282v1
    acc = asm.split("_", 2)
    if len(acc) < 2 or not asm.startswith("GC"):
        raise ValueError(f"Cannot derive FTP path from assembly name: {asm}")
    prefix, num = acc[0], acc[1].split(".")[0]  # GCF, 000012825
    if len(num) != 9 or not num.isdigit():
        raise ValueError(f"Unexpected accession numeric part in {asm}")
    return f"{NCBI_FTP_PREFIX}/{prefix}/{num[0:3]}/{num[3:6]}/{num[6:9]}/{asm}"


def load_species_accessions(path: Optional[Path] = None) -> dict[str, Any]:
    yaml_path = Path(path or DEFAULT_SPECIES_YAML)
    with open(yaml_path) as fh:
        return yaml.safe_load(fh) or {}


def genomes_for_cohort(cohort: str, species_cfg: Optional[Mapping[str, Any]] = None) -> dict[str, str]:
    """Return species -> assembly folder name for a named cohort set."""
    cfg = species_cfg or load_species_accessions()
    cohorts = cfg.get("cohorts") or {}
    if cohort not in cohorts:
        raise KeyError(f"Unknown species cohort '{cohort}'. Available: {sorted(cohorts)}")
    return dict(cohorts[cohort])


def ftp_list_for_cohort(cohort: str, species_cfg: Optional[Mapping[str, Any]] = None) -> dict[str, list[str]]:
    """Match calc_coverage.py historical shape: species -> [ftp_dir]."""
    return {sp: [accession_to_ftp(asm)] for sp, asm in genomes_for_cohort(cohort, species_cfg).items()}
