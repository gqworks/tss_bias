"""Bowtie2 mismatch-penalty sweep (Suppl. Fig. 7).

Penalties are 8, 6 (Bowtie2's default maximum), 4, and 2, passed as
``--mp {value}`` the way the lab scripts invoked Bowtie2. Mean TSS coverage
bias is recomputed for each penalty.
"""

from __future__ import annotations

import argparse

import pandas as pd

from tssbias.commands.common import passing_pairs, write_table
from tssbias.config import Config, Species
from tssbias.coverage import depth_from_bigwig, orient_windows
from tssbias.errors import require_paths
from tssbias.external import require_tool, run_command, run_pipeline
from tssbias.logutil import get_logger
from tssbias.metrics import coverage_bias
from tssbias.cohorts import fastq_paths
from tssbias.tss import read_tss_table

LOG = get_logger("tssbias.mismatch")


def run(cfg: Config, args: argparse.Namespace) -> None:
    """Align at each mismatch penalty and record mean bias.

    Outputs:
        ``results/mismatch/{cohort}_bias_by_mp.csv``
        BAMs ``{assembly}_{sample}_mp{value}.sorted.bam``
    """
    cohort = cfg.cohort(args.cohort)
    pairs = passing_pairs(cfg, cohort, args)
    require_tool("bowtie2")
    require_tool("samtools")
    require_tool("bamCoverage")
    rows = []
    for sample, species in pairs:
        reads = fastq_paths(cohort.fastq_dir, sample.sample_id, "illumina")
        require_paths(f"FASTQ for mismatch sweep {sample.sample_id}", reads)
        tss_path = cfg.tss_table_path(species.assembly)
        require_paths("filtered TSS table", [tss_path, cfg.fasta_path(species.assembly)])
        for penalty in cfg.bowtie2_mp:
            mean_bias, n_tss = _one_penalty(cfg, cohort.fastq_dir, sample.sample_id, species, penalty, args.force)
            rows.append(
                {
                    "sample": sample.sample_id,
                    "species": species.name,
                    "assembly": species.assembly,
                    "mismatch_penalty": penalty,
                    "mean_coverage_bias": mean_bias,
                    "n_tss": n_tss,
                }
            )
    shard_dir = cfg.results_dir / "mismatch" / cohort.name
    by_sample: dict[str, list] = {}
    for row in rows:
        by_sample.setdefault(row["sample"], []).append(row)
    for sample_id, sample_rows in by_sample.items():
        write_table(shard_dir / f"{sample_id}.csv", pd.DataFrame(sample_rows))
    if args.sample is not None or args.sample_index is not None:
        LOG.info("wrote per-sample mismatch tables under %s", shard_dir)
        return
    shards = [pd.read_csv(path) for path in sorted(shard_dir.glob("*.csv"))]
    destination = cfg.results_dir / "mismatch" / f"{cohort.name}_bias_by_mp.csv"
    write_table(destination, pd.concat(shards, ignore_index=True) if shards else pd.DataFrame(rows))
    LOG.info("wrote %s", destination)


def _one_penalty(cfg: Config, fastq_dir, sample: str, species: Species, penalty: int, force: bool) -> tuple[float, int]:
    tag = f"mp{penalty}"
    bam = cfg.bam_path(species.assembly, sample, tag)
    bigwig = cfg.bigwig_path(species.assembly, sample, tag)
    bam.parent.mkdir(parents=True, exist_ok=True)
    bigwig.parent.mkdir(parents=True, exist_ok=True)
    if force or not bam.exists():
        reads = fastq_paths(fastq_dir, sample, "illumina")
        fasta = cfg.fasta_path(species.assembly)
        threads = str(cfg.threads)
        run_pipeline(
            [
                [
                    "bowtie2",
                    "-p",
                    threads,
                    "--mp",
                    str(penalty),
                    "-x",
                    str(fasta),
                    "-1",
                    str(reads[0]),
                    "-2",
                    str(reads[1]),
                ],
                ["samtools", "view", "--threads", threads, "-b", "-"],
                ["samtools", "sort", "-@", threads, "-o", str(bam), "-"],
            ]
        )
        run_command(["samtools", "index", "-@", threads, str(bam)])
    if force or not bigwig.exists():
        run_command(
            ["bamCoverage", "--bam", str(bam), "-p", str(cfg.threads), "-o", str(bigwig), "-of", "bigwig"]
        )
    tss = read_tss_table(cfg.tss_table_path(species.assembly))
    windows, covered = orient_windows(depth_from_bigwig(bigwig), tss, cfg.window_bp)
    bias = coverage_bias(windows, cfg.window_bp)
    finite = bias[covered & pd.notna(bias)]
    if finite.size == 0:
        return float("nan"), 0
    return float(finite.mean()), int(finite.size)
