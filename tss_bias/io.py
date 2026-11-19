"""Sample-list and MetaPhlAn table loaders."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from tss_bias.paths import mpa_merged_path, reference_path, sample_list_path


def _read_list_lines(path: Path) -> list[str]:
    lines = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def load_sample_ids(cfg: Mapping[str, Any], list_name: str) -> list[str]:
    """Return first-column sample IDs from a sample list file."""
    path = sample_list_path(cfg, list_name)
    return [ln.split("\t")[0] for ln in _read_list_lines(path)]


def load_hmp2_pairs(cfg: Mapping[str, Any]) -> dict[str, str]:
    """dna_id -> rna_id for HMP2 matched MGX/MTX."""
    path = sample_list_path(cfg, cfg["cohorts"]["hmp2"]["sample_list"])
    out: dict[str, str] = {}
    for ln in _read_list_lines(path):
        parts = ln.split("\t")
        if len(parts) < 2:
            raise ValueError(f"HMP2 list expects dna_id\\trna_id, got: {ln}")
        out[parts[0]] = parts[1]
    return out


def cohort_fastq_dir(cfg: Mapping[str, Any], cohort: str) -> Path:
    c = cfg["cohorts"][cohort]
    return Path(c["fastq_dir"])


def load_sample_to_fastq_dir(cfg: Mapping[str, Any], cohort: str) -> dict[str, str]:
    """Map sample_id -> FASTQ directory using config cohort roots (no absolute cluster paths)."""
    c = cfg["cohorts"][cohort]
    ids = load_sample_ids(cfg, c["sample_list"])
    # For HMP2, sample list first column is DNA id
    root = str(Path(c["fastq_dir"]))
    return {sid: root for sid in ids}


def load_mpa_species_table(cfg: Mapping[str, Any], path: Optional[Path] = None) -> pd.DataFrame:
    """Load MetaPhlAn merged table, species-level rows only, indexed by s__Name."""
    mpa_path = Path(path) if path else mpa_merged_path(cfg)
    if not mpa_path.exists():
        raise FileNotFoundError(
            f"MetaPhlAn merged table not found: {mpa_path}\n"
            "Stage MetaPhlAn4 output as configured in paths.mpa_merged "
            "(see docs/DATA_STAGING.md). Bundled metaphlan3_res/ is incomplete "
            "and not a drop-in for mpa4_res/merged_mpa.txt."
        )
    mpa_res = pd.read_table(mpa_path, comment="#")
    mpa_res = mpa_res[
        (mpa_res["clade_name"].str.contains("s__")) & (~mpa_res["clade_name"].str.contains("t__"))
    ]
    mpa_res.index = [re.match(r".*\|([^\|]*)", x).group(1) for x in mpa_res["clade_name"]]
    return mpa_res.drop(columns=["clade_name"])


def filter_species_abundance(
    mpa: pd.DataFrame,
    sample_ids: list[str],
    min_pct: float = 0.5,
    min_samples: int = 5,
) -> pd.DataFrame:
    """Keep species with abundance > min_pct in at least min_samples; NaN below threshold."""
    cols = [c for c in sample_ids if c in mpa.columns]
    missing = set(sample_ids) - set(cols)
    if missing:
        raise KeyError(
            f"{len(missing)} samples missing from MetaPhlAn table (e.g. {next(iter(missing))}). "
            "Rebuild mpa4_res/merged_mpa.txt for this cohort."
        )
    sub = mpa.loc[:, cols]
    sub = sub.loc[((sub > min_pct).sum(axis=1) >= min_samples), :]
    return sub.where(sub > min_pct, np.nan)


def load_longread_abundance(cfg: Mapping[str, Any], platform: str, sample_ids: list[str]) -> pd.DataFrame:
    """Load bundled PacBio/ONT abundance tables from data/reference."""
    key = "pacbio_mpa" if platform == "pacbio" else "nanopore_mpa"
    path = reference_path(cfg, Path(cfg["reference_files"][key]).name)
    # Historical loader used custom names=sorted(samples)
    abund = pd.read_csv(path, sep="\t", header=0, index_col=0, names=sorted(sample_ids))
    abund = abund[(abund.index.str.contains("s__")) & (abund.index.str.contains("k__"))]
    abund.index = [re.match(r".*\|([^\|]*)", x).group(1) for x in abund.index]
    abund = abund[abund.mean(axis=1).sort_values(ascending=False) > 500]
    return 100 * abund / abund.sum(axis=0)
