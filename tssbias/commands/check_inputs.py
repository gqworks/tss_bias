"""Report which inputs a command will need, without writing results."""

from __future__ import annotations

import argparse
import shutil

from tssbias.cohorts import fastq_paths, read_manifest
from tssbias.commands.common import abundance_path
from tssbias.config import Config
from tssbias.refseq import resolved_table
from tssbias.logutil import get_logger

LOG = get_logger("tssbias.check-inputs")

TOOLS = {
    "discover": ["bowtie2", "bowtie2-build", "samtools", "prodigal"],
    "longread": ["bwa", "samtools", "prodigal"],
    "expression": ["kallisto", "samtools"],
    "mappability": ["art_illumina", "bowtie2", "bowtie2-build", "samtools"],
    "mismatch": ["bowtie2", "samtools", "bamCoverage"],
    "mag": ["megahit", "metabat2", "kraken2", "bowtie2", "samtools"],
    "promotech": ["python"],
    "sylph": ["sylph"],
    "matched": ["bowtie2", "prodigal", "minimap2", "samtools", "bamCoverage"],
}


def run(cfg: Config, args: argparse.Namespace) -> None:
    """Print a missing/present report for one cohort.

    Inputs:
        cfg, args: ``--cohort``.
    Outputs:
        Log lines. Exit status is 2 when a required file is missing, via
        SystemExit, so a scheduler can stop before a long job. This command
        does not create result files.
    """
    cohort = cfg.cohort(args.cohort)
    missing: list[str] = []

    def check(label: str, path) -> None:
        state = "ok" if path.exists() else "MISSING"
        LOG.info("%s %s %s", state, label, path)
        if state == "MISSING":
            missing.append(f"{label}: {path}")

    check("manifest", cohort.manifest)
    check("abundance", abundance_path(cfg, cohort))
    check("fastq_dir", cohort.fastq_dir)
    if cohort.manifest.exists():
        samples = read_manifest(
            cohort.manifest,
            sample_column=cohort.sample_column,
            longread_column=cohort.longread_column or None,
        )
        for sample in samples:
            for path in fastq_paths(cohort.fastq_dir, sample.sample_id, cohort.seq_type):
                check(f"fastq[{sample.sample_id}]", path)
            if cohort.paired_rna and sample.rna_id:
                rna = sample.rna_id
                check(f"rna[{rna}]_r1", cohort.fastq_dir / f"{rna}_1.fastq.gz")
                check(f"rna[{rna}]_r2", cohort.fastq_dir / f"{rna}_2.fastq.gz")
    if cohort.name != "mock":
        check("refseq_accessions", resolved_table(cfg.refseq_dir))
    toolset = "longread" if cohort.seq_type != "illumina" else "discover"
    for tool in TOOLS[toolset]:
        found = shutil.which(tool)
        LOG.info("%s tool %s %s", "ok" if found else "MISSING", tool, found or "")
        if not found:
            missing.append(f"tool: {tool}")
    if missing:
        LOG.info("cohort %s is not ready (%s missing)", cohort.name, len(missing))
        raise SystemExit(2)
    LOG.info("cohort %s has the discovery inputs that were checked", cohort.name)
