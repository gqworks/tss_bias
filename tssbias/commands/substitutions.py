"""Substitution counts (NM) by expression group at mismatch penalty 2.

Suppl. Fig. 8. Uniquely aligned reads are MAPQ >= 40. A read is counted in
the upstream flank when its start lies in
``[TSS - 500, TSS - read_length + buffer)`` on the plus strand (and the
strand-mirrored interval on the minus strand), so the alignment sits wholly
upstream. The downstream interval is mirrored into the gene. The mean NM tag
is the substitution count. Expression groups come from the expression command.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from tssbias.commands.common import passing_pairs, write_table
from tssbias.config import Config
from tssbias.errors import MissingInputError
from tssbias.logutil import get_logger
from tssbias.tss import read_tss_table

LOG = get_logger("tssbias.substitutions")


def run(cfg: Config, args: argparse.Namespace) -> None:
    """Summarise NM by expression group under ``--mp 2``.

    Outputs:
        ``results/substitutions/{cohort}_nm_by_gene.csv``
        ``results/substitutions/{cohort}_nm_by_group.csv``
        ``results/substitutions/{cohort}_tests.csv``
    """
    import pysam

    cohort = cfg.cohort(args.cohort)
    gene_table = cfg.results_dir / "expression" / f"{cohort.name}_gene_table.csv"
    if not gene_table.exists():
        raise MissingInputError(
            "substitution-by-expression needs the expression gene table",
            [gene_table],
        )
    groups = pd.read_csv(gene_table, usecols=["locus_tag", "sample", "species", "tpm_group"])
    pairs = passing_pairs(cfg, cohort, args)
    rows = []
    for sample, species in pairs:
        bam_path = cfg.bam_path(species.assembly, sample.sample_id, "mp2")
        tss_path = cfg.tss_table_path(species.assembly)
        if not bam_path.exists():
            raise MissingInputError(
                "mp2 BAM is missing. Run: python -m tssbias mismatch-penalty --cohort " + cohort.name,
                [bam_path],
            )
        tss = read_tss_table(tss_path)
        with pysam.AlignmentFile(str(bam_path), "rb") as bam:
            for row in tss.itertuples(index=False):
                rows.append(
                    {
                        "sample": sample.sample_id,
                        "species": species.name,
                        "locus_tag": row.locus_tag,
                        "upstream_nm": _mean_nm(bam, row, cfg, upstream=True),
                        "downstream_nm": _mean_nm(bam, row, cfg, upstream=False),
                    }
                )
    per_gene = pd.DataFrame(rows).merge(groups, on=["locus_tag", "sample", "species"], how="left")
    out = cfg.results_dir / "substitutions"
    write_table(out / f"{cohort.name}_nm_by_gene.csv", per_gene)
    summary = (
        per_gene.groupby(["sample", "species", "tpm_group"], as_index=False)[["upstream_nm", "downstream_nm"]]
        .mean()
    )
    write_table(out / f"{cohort.name}_nm_by_group.csv", summary)
    write_table(out / f"{cohort.name}_tests.csv", _tests(summary))
    LOG.info("wrote substitution tables for %s", cohort.name)


def _mean_nm(bam, row, cfg: Config, *, upstream: bool) -> float:
    start, end = _bounds(int(row.tss), str(row.strand), cfg.read_length, cfg.flank_buffer_bp, cfg.window_bp, upstream)
    if end <= start or start < 0:
        return float("nan")
    scores = []
    try:
        fetched = bam.fetch(str(row.genome), start, end)
    except ValueError:
        return float("nan")
    for read in fetched:
        if read.is_unmapped or read.mapping_quality < cfg.mapq_min:
            continue
        if read.reference_start < start or read.reference_start >= end:
            continue
        if read.has_tag("NM"):
            scores.append(float(read.get_tag("NM")))
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def _bounds(tss: int, strand: str, read_length: int, buffer: int, window_bp: int, upstream: bool) -> tuple[int, int]:
    """0-based half-open start interval used by the lab NM windows.

    Coordinates in the lab fetch were the 1-based TSS passed straight through.
    These bounds follow that spacing: a read-length gap keeps the alignment
    on one side of the TSS, with ``buffer`` bp of slack.
    """
    if strand == "+":
        if upstream:
            return tss - window_bp, tss - read_length + buffer
        return tss + read_length - buffer, tss + window_bp
    if upstream:
        return tss + read_length - buffer, tss + window_bp
    return tss - window_bp, tss - read_length + buffer


def _tests(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    order = ["zero", "low", "mid", "high"]
    for column in ("upstream_nm", "downstream_nm"):
        for left, right in zip(order, order[1:]):
            a = summary.loc[summary["tpm_group"] == left, column].dropna()
            b = summary.loc[summary["tpm_group"] == right, column].dropna()
            if len(a) < 2 or len(b) < 2:
                pvalue = float("nan")
            else:
                pvalue = float(ttest_ind(a, b).pvalue)
            rows.append({"flank": column, "contrast": f"{left}_vs_{right}", "pvalue": pvalue, "n_left": int(len(a)), "n_right": int(len(b))})
    return pd.DataFrame(rows)
