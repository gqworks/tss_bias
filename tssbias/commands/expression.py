"""Abundance-normalised expression groups versus TSS coverage bias.

kallisto TPM is divided by reads mapped to the reference:
``norm_tpm = 1e6 * tpm / mapped_reads``. Groups are zero / low (<100) /
mid (<1000) / high (>=1000) on ``norm_tpm``. ``log_norm_tpm`` uses the
natural log and pseudocount 0.001. GC bias is added when the reference FASTA
is present.
"""

from __future__ import annotations

import argparse
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, ttest_rel

from tssbias.commands.common import passing_pairs, write_table
from tssbias.config import Config
from tssbias.coverage import read_matrix
from tssbias.errors import MissingInputError, ToolError
from tssbias.external import require_tool, run_command
from tssbias.fasta import read_fasta
from tssbias.logutil import get_logger
from tssbias.metrics import (
    abundance_normalised_tpm,
    coverage_bias,
    expression_group,
    gc_bias_from_sequence,
)
from tssbias.tss import read_tss_table

LOG = get_logger("tssbias.expression")


def run(cfg: Config, args: argparse.Namespace) -> None:
    """Build the gene table, group means, paired tests, and a boxplot.

    Inputs:
        cfg, args: cohort selection. RNA FASTQs are ``{fastq_dir}/{rna_id}_1/_2``
        for cohorts with ``paired_rna``, otherwise the DNA id.
    Outputs:
        ``results/expression/{cohort}_gene_table.csv``
        ``results/expression/{cohort}_group_means.csv``
        ``results/expression/{cohort}_paired_tests.csv``
        ``results/expression/{cohort}_bias_by_group.png``
    """
    cohort = cfg.cohort(args.cohort)
    pairs = passing_pairs(cfg, cohort, args)
    frames = []
    for sample, species in pairs:
        frames.append(_one(cfg, cohort.fastq_dir, sample, species, cohort.paired_rna, args.force))
    genes = pd.concat(frames, ignore_index=True)
    out_dir = cfg.results_dir / "expression"
    write_table(out_dir / f"{cohort.name}_gene_table.csv", genes)
    grouped = (
        genes.dropna(subset=["coverage_bias"])
        .groupby(["species", "sample", "tpm_group"], as_index=False)["coverage_bias"]
        .mean()
    )
    write_table(out_dir / f"{cohort.name}_group_means.csv", grouped)
    tests = _paired_tests(grouped)
    stat, pvalue = _kruskal(grouped)
    tests = pd.concat(
        [tests, pd.DataFrame([{"contrast": "kruskal_four_groups", "n": len(grouped), "pvalue": pvalue, "statistic": stat}])],
        ignore_index=True,
    )
    write_table(out_dir / f"{cohort.name}_paired_tests.csv", tests)
    _boxplot(grouped, out_dir / f"{cohort.name}_bias_by_group.png", f"{cohort.name} coverage bias by expression")
    LOG.info("wrote expression tables for %s genes in %s", len(genes), cohort.name)


def _one(cfg: Config, fastq_dir, sample, species, paired_rna: bool, force: bool) -> pd.DataFrame:
    matrix = cfg.matrix_path(species.assembly, sample.sample_id)
    bam = cfg.bam_path(species.assembly, sample.sample_id)
    tss_path = cfg.tss_table_path(species.assembly)
    cds = cfg.assembly_dir(species.assembly) / f"{species.assembly}_prodigal.fna"
    missing = [path for path in (matrix, bam, tss_path, cds) if not path.exists()]
    if missing:
        raise MissingInputError(
            "expression needs the discovery matrix, BAM, filtered TSS table, and Prodigal CDS. "
            "Run prepare-references and discover first.",
            missing,
        )
    rna_id = sample.rna_id if paired_rna and sample.rna_id else sample.sample_id
    quant = cfg.kallisto_dir / f"{species.assembly}_{sample.sample_id}" / "abundance.tsv"
    if force or not quant.exists():
        r1 = fastq_dir / f"{rna_id}_1.fastq.gz"
        r2 = fastq_dir / f"{rna_id}_2.fastq.gz"
        if not r1.exists() or not r2.exists():
            raise MissingInputError(
                f"kallisto abundance is missing and RNA FASTQs for {rna_id} were not found",
                [quant, r1, r2],
            )
        _kallisto(cfg, cds, species.assembly, quant.parent, r1, r2)
    tpm = _read_tpm(quant)
    tags, windows = read_matrix(matrix, cfg.window_bp)
    bias = coverage_bias(windows, cfg.window_bp)
    covered = windows.sum(axis=1) > 0
    tss = read_tss_table(tss_path).drop_duplicates("locus_tag")
    frame = pd.DataFrame(
        {
            "locus_tag": tags.to_numpy(),
            "coverage_bias": bias,
            "covered": covered,
        }
    )
    frame = frame.merge(tss, on="locus_tag", how="left")
    frame["tpm"] = frame["locus_tag"].map(tpm)
    if frame["tpm"].notna().sum() == 0:
        raise MissingInputError(
            f"kallisto target_ids did not match Prodigal locus tags for {species.assembly}. "
            "No expression values were invented.",
            [quant],
        )
    mapped = _mapped_reads(bam)
    if mapped <= 0:
        raise MissingInputError(f"BAM {bam} has no mapped reads; cannot normalise TPM", [bam])
    norm, log_norm = abundance_normalised_tpm(frame["tpm"].to_numpy(), mapped, cfg.tpm_pseudocount)
    frame["mapped_reads"] = mapped
    frame["norm_tpm"] = norm
    frame["log_norm_tpm"] = log_norm
    frame["tpm_group"] = expression_group(norm, cfg.tpm_low, cfg.tpm_mid)
    frame["sample"] = sample.sample_id
    frame["species"] = species.name
    frame["assembly"] = species.assembly
    fasta = cfg.fasta_path(species.assembly)
    if fasta.exists():
        sequences = read_fasta(fasta)
        frame["gc_bias"] = [
            gc_bias_from_sequence(sequences[row.genome], int(row.tss), str(row.strand), cfg.window_bp)
            if isinstance(row.genome, str) and row.genome in sequences and pd.notna(row.tss)
            else np.nan
            for row in frame.itertuples(index=False)
        ]
    return frame


def _kallisto(cfg: Config, cds, assembly: str, out_dir, r1, r2) -> None:
    require_tool("kallisto")
    index = cfg.assembly_dir(assembly) / f"{assembly}_kallisto.idx"
    if not index.exists():
        run_command(["kallisto", "index", "-i", str(index), str(cds)])
    out_dir.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "kallisto",
            "quant",
            "-t",
            str(cfg.threads),
            "-i",
            str(index),
            "-o",
            str(out_dir),
            str(r1),
            str(r2),
        ]
    )


def _read_tpm(path) -> dict[str, float]:
    table = pd.read_csv(path, sep="\t")
    if "target_id" not in table.columns or "tpm" not in table.columns:
        raise MissingInputError("kallisto abundance.tsv needs target_id and tpm columns", [path])
    table = table.drop_duplicates("target_id")
    return dict(zip(table["target_id"].astype(str), table["tpm"].astype(float)))


def _mapped_reads(bam) -> int:
    require_tool("samtools")
    completed = subprocess.run(
        ["samtools", "idxstats", str(bam)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ToolError(f"samtools idxstats failed for {bam}: {completed.stderr.strip()}")
    total = 0
    for line in completed.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            total += int(parts[2])
    return total


def _paired_tests(grouped: pd.DataFrame) -> pd.DataFrame:
    order = ["zero", "low", "mid", "high"]
    rows = []
    for left, right in zip(order, order[1:]):
        a = grouped.loc[grouped["tpm_group"] == left, ["species", "sample", "coverage_bias"]]
        b = grouped.loc[grouped["tpm_group"] == right, ["species", "sample", "coverage_bias"]]
        merged = a.merge(b, on=["species", "sample"], suffixes=("_a", "_b"))
        if len(merged) < 2:
            pvalue = float("nan")
            stat = float("nan")
        else:
            result = ttest_rel(merged["coverage_bias_a"], merged["coverage_bias_b"])
            pvalue = float(result.pvalue)
            stat = float(result.statistic)
        rows.append({"contrast": f"{left}_vs_{right}", "n": int(len(merged)), "pvalue": pvalue, "statistic": stat})
    return pd.DataFrame(rows)


def _kruskal(grouped: pd.DataFrame) -> tuple[float, float]:
    samples = [
        grouped.loc[grouped["tpm_group"] == label, "coverage_bias"].dropna().to_numpy()
        for label in ("zero", "low", "mid", "high")
    ]
    if any(len(item) == 0 for item in samples):
        return float("nan"), float("nan")
    result = kruskal(*samples)
    return float(result.statistic), float(result.pvalue)


def _boxplot(grouped: pd.DataFrame, path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    order = ["zero", "low", "mid", "high"]
    data = [grouped.loc[grouped["tpm_group"] == label, "coverage_bias"].dropna().to_numpy() for label in order]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(data, tick_labels=order, showfliers=False)
    ax.set_xlabel("Expression group (abundance-normalised TPM)")
    ax.set_ylabel("Mean coverage bias")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
