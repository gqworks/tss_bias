"""Permutation test of mean TSS bias against rotated positions, then BH FDR.

Illumina uses 500 permutations. Long-read cohorts use 100. Pass
``--min-gene-bp 1000`` for the long-gene panel. ``--metric gc`` permutes GC
bias instead of coverage and does not need a bigWig.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tssbias.commands.common import passing_pairs, write_table
from tssbias.config import Config
from tssbias.coverage import depth_from_bigwig
from tssbias.errors import MissingInputError
from tssbias.fasta import read_fasta
from tssbias.logutil import get_logger
from tssbias.permute import permute_coverage, permute_gc
from tssbias.stats import benjamini_hochberg
from tssbias.tss import read_tss_table

LOG = get_logger("tssbias.permutation")


def run(cfg: Config, args: argparse.Namespace) -> None:
    """Write per-pair p-values and a cohort FDR table.

    Inputs:
        cfg: resolved config.
        args: cohort filters, ``--metric coverage|gc``, ``--min-gene-bp``,
            ``--fdr-only`` to combine files already written by an array job.
    Outputs:
        ``results/permutation/{cohort}_{metric}_pvalues.csv`` with raw p and
        BH FDR at the configured alpha. ``--fdr-only`` reads
        ``results/permutation/{cohort}_{metric}/``.
    """
    cohort = cfg.cohort(args.cohort)
    metric = args.metric
    tag = _tag(metric, args.min_gene_bp)
    out_dir = cfg.results_dir / "permutation" / f"{cohort.name}_{tag}"
    restricted = args.sample is not None or args.sample_index is not None
    if not args.fdr_only:
        pairs = passing_pairs(cfg, cohort, args)
        n_perm = cfg.longread_permutations if cohort.seq_type != "illumina" else cfg.n_permutations
        for sample, species in pairs:
            destination = out_dir / f"{species.assembly}_{sample.sample_id}.csv"
            if destination.exists() and not args.force:
                LOG.info("skip existing %s", destination)
                continue
            result = _one(cfg, cohort.seq_type, sample.sample_id, species.assembly, species.name, metric, n_perm, args.min_gene_bp)
            write_table(destination, pd.DataFrame([result]))
            LOG.info("pvalue %s %s %s = %s", cohort.name, species.name, sample.sample_id, result["pvalue"])
    if restricted and not args.fdr_only:
        LOG.info("per-pair p-values written; run permutation --fdr-only for the cohort BH adjustment")
        return
    combined = _combine(out_dir)
    if combined.empty:
        raise MissingInputError(
            f"no permutation results to adjust for {cohort.name} {tag}",
            [out_dir],
        )
    reject, adjusted = benjamini_hochberg(combined["pvalue"].to_numpy(), cfg.fdr_alpha)
    combined["fdr_bh"] = adjusted
    combined["significant_fdr_0.05" if cfg.fdr_alpha == 0.05 else "significant"] = reject
    destination = cfg.results_dir / "permutation" / f"{cohort.name}_{tag}_fdr.csv"
    write_table(destination, combined)
    n_sig = int(reject.sum())
    LOG.info("wrote %s (%s / %s FDR < %s)", destination, n_sig, len(combined), cfg.fdr_alpha)


def _one(cfg: Config, seq_type: str, sample: str, assembly: str, species: str, metric: str, n_perm: int, min_gene_bp: int) -> dict:
    del seq_type
    tss_path = cfg.tss_table_path(assembly)
    if not tss_path.exists():
        raise MissingInputError("filtered TSS table", [tss_path])
    tss = read_tss_table(tss_path)
    if metric == "coverage":
        bigwig = cfg.bigwig_path(assembly, sample)
        if not bigwig.exists():
            raise MissingInputError(
                "coverage bigWig. Run python -m tssbias discover for this cohort first.",
                [bigwig],
            )
        result = permute_coverage(
            depth_from_bigwig(bigwig),
            tss,
            n_permutations=n_perm,
            seed=cfg.seed,
            sample=sample,
            assembly=assembly,
            window_bp=cfg.window_bp,
            min_gene_bp=min_gene_bp,
        )
    elif metric == "gc":
        fasta = cfg.fasta_path(assembly)
        if not fasta.exists():
            raise MissingInputError("reference FASTA for GC permutation", [fasta])
        result = permute_gc(
            read_fasta(fasta),
            tss,
            n_permutations=n_perm,
            seed=cfg.seed,
            sample=sample,
            assembly=assembly,
            window_bp=cfg.window_bp,
        )
    else:
        raise MissingInputError(f"unknown permutation metric {metric}", [])
    return {
        "sample": sample,
        "species": species,
        "assembly": assembly,
        "metric": metric,
        "min_gene_bp": min_gene_bp,
        "n_tss": result.n_tss,
        "n_permutations": result.n_permutations,
        "mean_statistic": result.mean_statistic,
        "null_mean": result.null_mean,
        "pvalue": result.pvalue,
    }


def _combine(directory: Path) -> pd.DataFrame:
    if not directory.exists():
        return pd.DataFrame()
    frames = [pd.read_csv(path) for path in sorted(directory.glob("*.csv"))]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _tag(metric: str, min_gene_bp: int) -> str:
    if min_gene_bp > 0:
        return f"{metric}_genelen{min_gene_bp}"
    return metric
