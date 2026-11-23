"""Primary discovery path: align reads and write ±500 bp TSS coverage matrices.

One sample is one alignment. Bowtie2 (Illumina) or BWA (long read) maps that
sample to a concatenated FASTA of the species that pass in the sample. Each
read is assigned to a single best genome. The alignment is split into the
per-species BAMs later commands read. Unmapped reads are dropped.
``samtools depth -aa`` supplies per-base depth, including contigs with no
reads, for the ±500 bp CSV and for permutation. References are whole genomes,
not TSS windows.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

from tssbias.commands.common import abundance_path, passing_pairs, selected_samples
from tssbias.concat import rewrite_alignment, rewrite_sq, write_concat_fasta
from tssbias.cohorts import fastq_paths
from tssbias.config import Config, Species
from tssbias.coverage import depth_from_table, orient_windows, write_matrix
from tssbias.errors import MissingInputError, ToolError, require_paths
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
        Per-species BAMs and ``samtools depth -aa`` tables are written under the
        scratch paths. A read maps to one best genome in the concatenated
        reference, not to every genome it would have matched alone. Missing
        FASTQs, abundance tables, or reference FASTAs raise MissingInputError
        before any matrix is written.
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
    grouped: dict[str, list[Species]] = {}
    order: list[str] = []
    for sample, species in pairs:
        if sample.sample_id not in grouped:
            order.append(sample.sample_id)
            grouped[sample.sample_id] = []
        grouped[sample.sample_id].append(species)
    for sample_id in order:
        discover_sample(
            cfg,
            cohort.seq_type,
            cohort.fastq_dir,
            sample_id,
            grouped[sample_id],
            force=args.force,
        )
    LOG.info("discovery wrote matrices for %s samples in cohort %s", len(order), cohort.name)


def _require_aligner(seq_type: str) -> None:
    require_tool("samtools")
    if seq_type == "illumina":
        require_tool("bowtie2")
        require_tool("bowtie2-build")
    elif seq_type in {"pacbio", "nanopore"}:
        require_tool("bwa")
    else:
        raise MissingInputError(f"unknown seq_type {seq_type}", [])


def _one_pair(cfg: Config, seq_type: str, fastq_dir, sample: str, species: Species, *, force: bool) -> None:
    """Align one sample to one reference. Used by mock and the mappability control."""
    discover_sample(cfg, seq_type, fastq_dir, sample, [species], force=force)


def discover_sample(
    cfg: Config,
    seq_type: str,
    fastq_dir,
    sample: str,
    species: list[Species],
    *,
    force: bool,
) -> None:
    """Align one sample once, then write each species BAM, depth table, and matrix.

    A sample whose matrices already exist is skipped unless ``force`` is set.
    When any matrix is missing, every species for that sample is aligned
    together so the best-genome assignment is one decision.
    """
    if not species:
        return
    matrices = [cfg.matrix_path(item.assembly, sample) for item in species]
    if not force and all(path.exists() for path in matrices):
        LOG.info("skip %s; %s matrices exist", sample, len(matrices))
        return
    reads = fastq_paths(fastq_dir, sample, seq_type)
    require_paths(f"FASTQ for {sample}", reads)
    concat = _concat_reference(cfg, seq_type, species)
    combined = cfg.scratch_root / "discover_concat" / "align" / f"{sample}.sorted.bam"
    combined.parent.mkdir(parents=True, exist_ok=True)
    _align_concat(cfg, seq_type, reads, concat, combined)
    destinations = {item.assembly: cfg.bam_path(item.assembly, sample) for item in species}
    _split_bam(combined, list(destinations), destinations, str(cfg.threads))
    combined.unlink(missing_ok=True)
    for item in species:
        bam = destinations[item.assembly]
        if not bam.with_name(bam.name + ".bai").exists():
            run_command(["samtools", "index", "-@", str(cfg.threads), str(bam)])
        depth_path = cfg.depth_path(item.assembly, sample)
        _write_depth(bam, depth_path, str(cfg.threads))
        tss = read_tss_table(cfg.tss_table_path(item.assembly))
        windows, _covered = orient_windows(depth_from_table(depth_path), tss, cfg.window_bp)
        matrix = cfg.matrix_path(item.assembly, sample)
        write_matrix(matrix, tss["locus_tag"].astype(str).tolist(), windows, cfg.window_bp)
        LOG.info("wrote %s", matrix)


def _concat_reference(cfg: Config, seq_type: str, species: list[Species]) -> Path:
    names = sorted(item.assembly for item in species)
    digest = hashlib.sha256("\n".join(names).encode()).hexdigest()[:16]
    directory = cfg.scratch_root / "discover_concat" / digest
    fasta = directory / "concat.fna"
    sources = [(item.assembly, cfg.fasta_path(item.assembly)) for item in sorted(species, key=lambda item: item.assembly)]
    if not fasta.exists():
        write_concat_fasta(sources, fasta)
    if seq_type == "illumina":
        indexed = (directory / "concat.fna.1.bt2").exists() or (directory / "concat.fna.1.bt2l").exists()
        if not indexed:
            run_command(["bowtie2-build", str(fasta), str(fasta)])
    elif not (directory / "concat.fna.bwt").exists():
        run_command(["bwa", "index", str(fasta)])
    return fasta


def _align_concat(cfg: Config, seq_type: str, reads: list[Path], concat: Path, bam: Path) -> None:
    threads = str(cfg.threads)
    if seq_type == "illumina":
        align = ["bowtie2", "--no-unal", "-p", threads, "-x", str(concat), "-1", str(reads[0]), "-2", str(reads[1])]
    else:
        preset = "pacbio" if seq_type == "pacbio" else "ont2d"
        align = ["bwa", "mem", "-t", threads, "-x", preset, str(concat), str(reads[0])]
    view = ["samtools", "view", "--threads", threads, "-F", "4", "-b", "-"]
    sort = ["samtools", "sort", "-@", threads, "-o", str(bam), "-"]
    run_pipeline([align, view, sort])


def _write_depth(bam: Path, destination: Path, threads: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    run_command(["samtools", "depth", "-aa", "-@", threads, "-o", str(partial), str(bam)])
    partial.replace(destination)


def _split_bam(concat_bam: Path, assemblies: list[str], destinations: dict[str, Path], threads: str) -> None:
    """Split a prefixed BAM into one coordinate-sorted BAM per assembly."""
    view = subprocess.Popen(
        ["samtools", "view", "-h", str(concat_bam)],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert view.stdout is not None
    header: list[str] = []
    sq: dict[str, list[str]] = {assembly: [] for assembly in assemblies}
    writers: dict[str, subprocess.Popen[str]] = {}
    try:
        for line in view.stdout:
            if line.startswith("@"):
                if line.startswith("@SQ"):
                    for assembly in assemblies:
                        rewritten = rewrite_sq(line, assembly)
                        if rewritten is not None:
                            sq[assembly].append(rewritten)
                            break
                else:
                    header.append(line)
                continue
            if not writers:
                writers = _open_sorts(assemblies, destinations, header, sq, threads)
            fields = line.rstrip("\n").split("\t")
            for assembly, proc in writers.items():
                rewritten = rewrite_alignment(fields, assembly)
                if rewritten is not None:
                    assert proc.stdin is not None
                    proc.stdin.write("\t".join(rewritten) + "\n")
                    break
        if not writers:
            writers = _open_sorts(assemblies, destinations, header, sq, threads)
    finally:
        if view.stdout is not None:
            view.stdout.close()
        view.wait()
        for proc in writers.values():
            if proc.stdin is not None:
                proc.stdin.close()
        codes = [proc.wait() for proc in writers.values()]
    if view.returncode not in (0, None):
        raise ToolError(f"samtools view exited {view.returncode}")
    failed = [code for code in codes if code != 0]
    if failed:
        raise ToolError(f"samtools sort exited {failed[0]} while splitting {concat_bam.name}")


def _open_sorts(
    assemblies: list[str],
    destinations: dict[str, Path],
    header: list[str],
    sq: dict[str, list[str]],
    threads: str,
) -> dict[str, subprocess.Popen[str]]:
    writers: dict[str, subprocess.Popen[str]] = {}
    for assembly in assemblies:
        dest = destinations[assembly]
        dest.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(
            ["samtools", "sort", "-@", threads, "-o", str(dest), "-"],
            stdin=subprocess.PIPE,
            text=True,
        )
        assert proc.stdin is not None
        for line in header:
            proc.stdin.write(line)
        for line in sq[assembly]:
            proc.stdin.write(line)
        writers[assembly] = proc
    return writers
