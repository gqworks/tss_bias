"""Resolve RefSeq genomes for screened species, call Prodigal, and write TSS tables.

The species list is the cohort MetaPhlAn screen. NCBI supplies the reference
genome, or the representative genome when there is no reference. TSS sites
more than 500 bp from neighbouring gene starts are written to
``{assembly}_filtered_tss.csv``. Bowtie2 (Illumina) and BWA (long-read)
indexes are built when the cohort needs them.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import urllib.request
from pathlib import Path

from tssbias.commands.common import screened_names
from tssbias.config import Config, Species
from tssbias.errors import MissingInputError
from tssbias.external import require_tool, run_command
from tssbias.logutil import get_logger
from tssbias.refseq import lookup_refseq, read_resolved, resolved_table, write_resolved
from tssbias.tss import filter_isolated_tss, parse_prodigal_gff

LOG = get_logger("tssbias.prepare-references")


def run(cfg: Config, args: argparse.Namespace) -> None:
    """Resolve and prepare every species that passes the cohort screen.

    Inputs:
        cfg: resolved config.
        args: ``--cohort`` and optional ``--species``.
    Outputs:
        ``resolved_accessions.tsv``, FASTA, Prodigal GFF, filtered TSS CSV,
        and an aligner index under ``refseq_dir``. A species with no RefSeq
        reference or representative genome is recorded as unresolved and
        skipped. Raises MissingInputError if the FTP download fails.
    """
    cohort = cfg.cohort(args.cohort)
    names = screened_names(cfg, cohort, args)
    index = "bwa" if cohort.seq_type in {"pacbio", "nanopore"} else "bowtie2"
    cache_path = resolved_table(cfg.refseq_dir)
    cached = read_resolved(cache_path)
    prepared = 0
    for name in names:
        current = cached.get(name)
        if args.force or current is None:
            current = lookup_refseq(name)
            cached[name] = current
            write_resolved(cache_path, cached)
        if not current.assembly:
            LOG.info("no RefSeq reference or representative genome for %s", name)
            continue
        _prepare_one(cfg, current, index=index, window_bp=cfg.window_bp, force=args.force)
        prepared += 1
    LOG.info("prepared %s assemblies for cohort %s (%s screened)", prepared, cohort.name, len(names))


def _prepare_one(cfg: Config, species: Species, *, index: str, window_bp: int, force: bool) -> None:
    directory = cfg.assembly_dir(species.assembly)
    directory.mkdir(parents=True, exist_ok=True)
    fasta = cfg.fasta_path(species.assembly)
    if force or not fasta.exists():
        _download_fasta(species, fasta)
    gff = directory / f"{species.assembly}_prodigal.gff"
    cds = directory / f"{species.assembly}_prodigal.fna"
    if force or not gff.exists() or not cds.exists():
        require_tool("prodigal")
        run_command(
            [
                "prodigal",
                "-i",
                str(fasta),
                "-o",
                str(gff),
                "-d",
                str(cds),
                "-f",
                "gff",
            ]
        )
    table = cfg.tss_table_path(species.assembly)
    if force or not table.exists():
        filtered = filter_isolated_tss(parse_prodigal_gff(gff), window_bp)
        filtered.to_csv(table, index=False)
        LOG.info("wrote %s TSS sites to %s", len(filtered), table)
    if index == "bowtie2":
        if force or not (directory / f"{species.assembly}.fna.1.bt2").exists():
            require_tool("bowtie2-build")
            run_command(["bowtie2-build", str(fasta), str(fasta)])
    elif force or not (directory / f"{species.assembly}.fna.bwt").exists():
        require_tool("bwa")
        run_command(["bwa", "index", str(fasta)])


def _download_fasta(species: Species, destination: Path) -> None:
    url = f"{species.ftp}/{species.assembly}_genomic.fna.gz"
    gzip_path = destination.with_suffix(".fna.gz")
    LOG.info("download %s", url)
    try:
        urllib.request.urlretrieve(url, gzip_path)
    except Exception as exc:  # network and HTTP errors
        raise MissingInputError(
            f"could not download {species.assembly} ({exc}). "
            "Place the gunzipped FASTA at the path below, or retry on a machine with FTP access.",
            [destination],
        ) from exc
    with gzip.open(gzip_path, "rb") as src, destination.open("wb") as dest:
        shutil.copyfileobj(src, dest)
    gzip_path.unlink(missing_ok=True)
