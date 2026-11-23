"""Primary discovery path: align reads and write ±500 bp TSS coverage matrices.

This is the Illumina analysis that the lab scripts left commented out. One
sample-species pair is one task, so a PBS array can pass ``--sample-index``.
Long-read cohorts use ``bwa mem -x pacbio`` or ``-x ont2d`` instead of Bowtie2.
"""

from __future__ import annotations

import argparse

from tssbias.commands.common import abundance_path, passing_pairs, selected_samples
from tssbias.cohorts import fastq_paths
from tssbias.config import Config, Species
from tssbias.coverage import depth_from_bigwig, orient_windows, write_matrix
from tssbias.errors import MissingInputError, require_paths
from tssbias.external import require_tool, run_command, run_pipeline
from tssbias.logutil import get_logger
from tssbias.tss import read_tss_table

LOG = get_logger("tssbias.discover")


def run(cfg: Config, args: argparse.Namespace) -> None:
    """Align one cohort and write a coverage matrix per passing pair.

    Inputs:
        cfg: resolved config.
        args: cohort, optional sample / sample-index / species / force.
    Outputs:
        ``results/matrices/{assembly}_{sample}.csv`` with 1001 coverage columns.
        BAMs and bigWigs are written under the scratch paths. Missing FASTQs,
        abundance tables, or reference FASTAs raise MissingInputError before
        any matrix is written for that pair.
    """
    cohort = cfg.cohort(args.cohort)
    samples = selected_samples(cfg, cohort, args)
    staged = []
    table = abundance_path(cfg, cohort)
    if not table.exists():
        staged.append(table)
    for sample in samples:
        for path in fastq_paths(cohort.fastq_dir, sample.sample_id, cohort.seq_type):
            if not path.exists():
                staged.append(path)
    if staged:
        raise MissingInputError(
            f"cohort {cohort.name} FASTQs must already be staged under {cohort.fastq_dir} "
            "as {sample_id}_1.fastq.gz and {sample_id}_2.fastq.gz (Illumina) "
            "or {sample_id}.fastq.gz (long read). This command does not download FASTQs.",
            staged,
        )
    pairs = passing_pairs(cfg, cohort, args)
    missing_ref = []
    for sample, species in pairs:
        if not cfg.fasta_path(species.assembly).exists() or not cfg.tss_table_path(species.assembly).exists():
            missing_ref.append(cfg.fasta_path(species.assembly))
            missing_ref.append(cfg.tss_table_path(species.assembly))
    if missing_ref:
        raise MissingInputError(
            f"reference FASTA or filtered TSS table is missing for cohort {cohort.name}. "
            "Run: python -m tssbias prepare-references --cohort " + cohort.name,
            sorted(set(missing_ref)),
        )
    _require_aligner(cohort.seq_type)
    for sample, species in pairs:
        _one_pair(
            cfg,
            cohort.seq_type,
            cohort.fastq_dir,
            sample.sample_id,
            species,
            force=args.force,
        )
    LOG.info("discovery wrote %s matrices for cohort %s", len(pairs), cohort.name)


def _require_aligner(seq_type: str) -> None:
    require_tool("samtools")
    require_tool("bamCoverage")
    if seq_type == "illumina":
        require_tool("bowtie2")
    elif seq_type in {"pacbio", "nanopore"}:
        require_tool("bwa")
    else:
        raise MissingInputError(f"unknown seq_type {seq_type}", [])


def _one_pair(cfg: Config, seq_type: str, fastq_dir, sample: str, species: Species, *, force: bool) -> None:
    matrix = cfg.matrix_path(species.assembly, sample)
    if matrix.exists() and not force:
        LOG.info("skip existing matrix %s", matrix)
        return
    bam = cfg.bam_path(species.assembly, sample)
    bigwig = cfg.bigwig_path(species.assembly, sample)
    bam.parent.mkdir(parents=True, exist_ok=True)
    bigwig.parent.mkdir(parents=True, exist_ok=True)
    if force or not bam.exists():
        _align(cfg, seq_type, fastq_dir, sample, species)
    if not bam.with_name(bam.name + ".bai").exists():
        run_command(["samtools", "index", "-@", str(cfg.threads), str(bam)])
    if force or not bigwig.exists():
        run_command(
            [
                "bamCoverage",
                "--bam",
                str(bam),
                "-p",
                str(cfg.threads),
                "-o",
                str(bigwig),
                "-of",
                "bigwig",
            ]
        )
    tss = read_tss_table(cfg.tss_table_path(species.assembly))
    depth = depth_from_bigwig(bigwig)
    windows, _covered = orient_windows(depth, tss, cfg.window_bp)
    write_matrix(matrix, tss["locus_tag"].astype(str).tolist(), windows, cfg.window_bp)
    LOG.info("wrote %s", matrix)


def _align(cfg: Config, seq_type: str, fastq_dir, sample: str, species: Species) -> None:
    reads = fastq_paths(fastq_dir, sample, seq_type)
    require_paths(f"FASTQ for {sample}", reads)
    fasta = cfg.fasta_path(species.assembly)
    bam = cfg.bam_path(species.assembly, sample)
    threads = str(cfg.threads)
    if seq_type == "illumina":
        align = [
            "bowtie2",
            "-p",
            threads,
            "-x",
            str(fasta),
            "-1",
            str(reads[0]),
            "-2",
            str(reads[1]),
        ]
    else:
        preset = "pacbio" if seq_type == "pacbio" else "ont2d"
        align = ["bwa", "mem", "-t", threads, "-x", preset, str(fasta), str(reads[0])]
    view = ["samtools", "view", "--threads", threads, "-b", "-"]
    sort = ["samtools", "sort", "-@", threads, "-o", str(bam), "-"]
    run_pipeline([align, view, sort])
