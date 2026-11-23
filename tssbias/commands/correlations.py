"""Pairwise Pearson correlation of per-TSS bias vectors.

Cohort replication uses species that pass 0.5% abundance in at least 5 samples
of every cohort, and only the first 20 iHMP samples so the cohorts are closer
in size. The other Illumina cohorts are ``liu2016`` (Liu et al., all 110
public runs) and ``hmp3_stool`` (every HMP Phase III stool specimen that
survives the one-run-per-alias rule).
Platform comparison uses the long-read species set, with the iHMP
threshold at 30 samples and long-read thresholds at 4.
"""

from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from tssbias.cohorts import abundant_pairs, read_abundance, read_manifest
from tssbias.commands.common import write_table
from tssbias.config import Config
from tssbias.coverage import read_matrix
from tssbias.errors import MissingInputError
from tssbias.logutil import get_logger
from tssbias.metrics import coverage_bias

LOG = get_logger("tssbias.correlation")


def run_cohorts(cfg: Config, args: argparse.Namespace) -> None:
    """Within- and across-cohort correlations for the Illumina replication set.

    Outputs:
        ``results/correlation/cohort_pairwise_r.csv``
        ``results/correlation/cohort_summary.csv``
    """
    del args
    names = ["hmp2", "franzosa", "liu2016", "hmp3_stool"]
    _correlate(cfg, names, cfg.species_in("correlation"), balance={"hmp2": cfg.correlation_balance_n}, stem="cohort")


def run_platforms(cfg: Config, args: argparse.Namespace) -> None:
    """Illumina vs PacBio vs Nanopore correlations.

    Outputs:
        ``results/correlation/platform_pairwise_r.csv``
        ``results/correlation/platform_summary.csv``
    """
    del args
    _correlate(cfg, ["hmp2", "pacbio", "nanopore"], cfg.species_in("longread"), balance={}, stem="platform")


def _correlate(cfg: Config, cohort_names: list[str], species: list, balance: dict[str, int], stem: str) -> None:
    vectors: list[dict] = []
    for name in cohort_names:
        cohort = cfg.cohort(name)
        samples = read_manifest(cohort.manifest)
        if name in balance:
            samples = samples[: balance[name]]
        abundance = read_abundance(cohort.abundance_table or cfg.mpa_merged)
        mask = abundant_pairs(
            abundance,
            read_manifest(cohort.manifest),
            [item.name for item in species],
            cohort.abundance_min_pct,
            cohort.abundance_min_samples,
        )
        for item in species:
            if item.name not in mask.index:
                continue
            for sample in samples:
                if sample.sample_id not in mask.columns or not bool(mask.at[item.name, sample.sample_id]):
                    continue
                matrix = cfg.matrix_path(item.assembly, sample.sample_id)
                if not matrix.exists():
                    raise MissingInputError(
                        f"coverage matrix missing for {name} {item.name} {sample.sample_id}. Run discover first.",
                        [matrix],
                    )
                tags, windows = read_matrix(matrix, cfg.window_bp)
                bias = coverage_bias(windows, cfg.window_bp)
                vectors.append(
                    {
                        "cohort": name,
                        "species": item.name,
                        "sample": sample.sample_id,
                        "tags": tags.to_numpy(),
                        "bias": bias,
                    }
                )
    if len(vectors) < 2:
        raise MissingInputError("need at least two bias vectors to correlate", [cfg.matrices_dir])
    rows = []
    for left, right in combinations(vectors, 2):
        if left["species"] != right["species"]:
            continue
        shared = _align(left, right)
        if shared is None:
            continue
        a, b = shared
        if np.std(a) == 0 or np.std(b) == 0:
            coefficient, pvalue = float("nan"), float("nan")
        else:
            coefficient, pvalue = pearsonr(a, b)
        rows.append(
            {
                "species": left["species"],
                "cohort_a": left["cohort"],
                "sample_a": left["sample"],
                "cohort_b": right["cohort"],
                "sample_b": right["sample"],
                "n_tss": int(len(a)),
                "pearson_r": float(coefficient),
                "pvalue": float(pvalue),
                "same_cohort": left["cohort"] == right["cohort"],
            }
        )
    pairwise = pd.DataFrame(rows)
    out = cfg.results_dir / "correlation"
    write_table(out / f"{stem}_pairwise_r.csv", pairwise)
    if pairwise.empty:
        raise MissingInputError("no shared TSS vectors produced a correlation", [cfg.matrices_dir])
    summary = (
        pairwise.groupby(["species", "cohort_a", "cohort_b"], as_index=False)["pearson_r"].mean()
    )
    write_table(out / f"{stem}_summary.csv", summary)
    LOG.info("wrote %s correlations to %s", len(pairwise), out)


def _align(left: dict, right: dict):
    order = {tag: index for index, tag in enumerate(right["tags"])}
    a = []
    b = []
    for tag, value in zip(left["tags"], left["bias"]):
        index = order.get(tag)
        if index is None or not np.isfinite(value) or not np.isfinite(right["bias"][index]):
            continue
        a.append(value)
        b.append(right["bias"][index])
    if len(a) < 3:
        return None
    return np.asarray(a), np.asarray(b)
