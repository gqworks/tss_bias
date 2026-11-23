"""Download SRA runs listed in a manifest.

Gadi users can wrap this command in a PBS job. It does not submit job
arrays, retry failed dumps, or download runs in parallel.

``python -m tssbias download data/manifests/hmp2.tsv``
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import tempfile
from pathlib import Path

from tssbias.cohorts import read_manifest
from tssbias.config import Config
from tssbias.errors import ToolError
from tssbias.external import require_tool, run_command
from tssbias.logutil import get_logger

LOG = get_logger("tssbias.download")


def run(cfg: Config, args: argparse.Namespace) -> None:
    """Download ``sample_id`` runs, plus extra accessions when present.

    Inputs:
        args.manifest: TSV with a ``sample_id`` column. The output directory
            name is the filename stem (``hmp2.tsv`` → ``fastq/hmp2``).
    Outputs:
        ``{data_root}/fastq/<stem>/{SRR}_1.fastq.gz`` and ``_2`` for paired
        reads. ``rna_id`` goes in ``rna/``. ``longread_srr`` goes in
        ``longread/``. ``shortread_srr`` goes in ``shortread/``. A dump that
        emits one FASTQ is kept as that one file.
    """
    manifest = Path(args.manifest)
    require_tool("fasterq-dump")
    samples = read_manifest(manifest)
    out = cfg.data_root / "fastq" / manifest.stem
    for sample in samples:
        _fetch(sample.sample_id, out)
        if sample.rna_id:
            _fetch(sample.rna_id, out / "rna")
        if sample.longread_id:
            _fetch(sample.longread_id, out / "longread")
        if sample.shortread_id:
            _fetch(sample.shortread_id, out / "shortread")
    LOG.info("manifest %s: %s samples under %s", manifest.name, len(samples), out)


def _fetch(srr: str, dest: Path) -> None:
    if _present(dest, srr):
        LOG.info("skip %s; FASTQ already in %s", srr, dest)
        return
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{srr}.", dir=dest) as tmp_name:
        tmp = Path(tmp_name)
        run_command(["fasterq-dump", "--split-files", "-O", str(tmp), srr])
        _publish(srr, tmp, dest)


def _present(dest: Path, srr: str) -> bool:
    single = dest / f"{srr}.fastq.gz"
    mate1 = dest / f"{srr}_1.fastq.gz"
    mate2 = dest / f"{srr}_2.fastq.gz"
    if single.exists():
        return True
    if mate1.exists() and mate2.exists():
        return True
    # A finished single-end dump kept under the emitted ``_1`` name.
    return mate1.exists() and not mate2.exists()


def _publish(srr: str, tmp: Path, dest: Path) -> None:
    mate1 = tmp / f"{srr}_1.fastq"
    mate2 = tmp / f"{srr}_2.fastq"
    single = tmp / f"{srr}.fastq"
    if mate1.is_file() and mate2.is_file():
        _gzip_pair(mate1, dest / f"{srr}_1.fastq.gz", mate2, dest / f"{srr}_2.fastq.gz")
        return
    if single.is_file():
        _gzip_one(single, dest / f"{srr}.fastq.gz")
        return
    if mate1.is_file():
        _gzip_one(mate1, dest / f"{srr}_1.fastq.gz")
        return
    found = sorted(path.name for path in tmp.iterdir())
    raise ToolError(f"fasterq-dump wrote no FASTQ for {srr}. Found: {found or 'nothing'}")


def _gzip_one(src: Path, dest: Path) -> None:
    partial = dest.with_name(dest.name + ".partial")
    with src.open("rb") as raw, gzip.open(partial, "wb") as out:
        shutil.copyfileobj(raw, out, length=1024 * 1024)
    partial.replace(dest)


def _gzip_pair(src1: Path, dest1: Path, src2: Path, dest2: Path) -> None:
    partial1 = dest1.with_name(dest1.name + ".partial")
    partial2 = dest2.with_name(dest2.name + ".partial")
    for src, partial in ((src1, partial1), (src2, partial2)):
        with src.open("rb") as raw, gzip.open(partial, "wb") as out:
            shutil.copyfileobj(raw, out, length=1024 * 1024)
    partial1.replace(dest1)
    partial2.replace(dest2)
