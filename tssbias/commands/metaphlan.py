"""Run MetaPhlAn 4 and merge species profiles.

``python -m tssbias metaphlan data/manifests/hmp2.tsv``

Reads the FASTQs already staged for that manifest and writes
``{data_root}/mpa4_res/<cohort>/merged_mpa.txt``. The ``metaphlan`` binary
is the MetaPhlAn 4.1.0 pin in ``environment.yml``. This command does not
download reads and does not use a checked-in profile.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tssbias.cohorts import Sample, fastq_paths, read_manifest
from tssbias.commands.common import abundance_path
from tssbias.config import Cohort, Config
from tssbias.errors import MissingInputError
from tssbias.external import require_tool, run_command
from tssbias.logutil import get_logger

LOG = get_logger("tssbias.metaphlan")


def run(cfg: Config, args: argparse.Namespace) -> None:
    """Profile staged FASTQs and write the merged species table.

    Inputs:
        args.manifest: TSV with ``sample_id``. When it is a cohort manifest,
            FASTQs are read from that cohort's ``fastq_dir``.
    Outputs:
        Per-sample profiles and ``merged_mpa.txt`` under
        ``{data_root}/mpa4_res/<cohort>/``. That file is the cohort
        ``abundance_table``. ``--sample-index`` writes one profile and
        does not merge.
    """
    manifest = _resolve_manifest(cfg, Path(args.manifest))
    cohort = _cohort_for_manifest(cfg, manifest)
    if cohort is None:
        name = manifest.stem
        fastq_dir = cfg.data_root / "fastq" / name
        seq_type = _seq_type(manifest)
        samples = read_manifest(manifest)
        destination = cfg.data_root / "mpa4_res" / name / "merged_mpa.txt"
    else:
        name = cohort.name
        fastq_dir = cohort.fastq_dir
        seq_type = cohort.seq_type
        samples = read_manifest(cohort.manifest)
        destination = abundance_path(cfg, cohort)
    selected = _select(samples, manifest, args)
    missing = _missing_fastqs(selected, fastq_dir, seq_type)
    if missing:
        shown = manifest
        try:
            shown = manifest.relative_to(cfg.repo_root)
        except ValueError:
            pass
        raise MissingInputError(
            f"FASTQs for MetaPhlAn must already be staged under {fastq_dir}. "
            "This command does not download FASTQs. "
            f"Run: python -m tssbias download {shown}",
            missing,
        )
    require_tool("metaphlan")
    out_dir = cfg.data_root / "mpa4_res" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    for sample in selected:
        profile = out_dir / f"{sample.sample_id}.txt"
        if args.force or not profile.exists():
            reads = fastq_paths(fastq_dir, sample.sample_id, seq_type)
            _profile(cfg, reads, profile)
        LOG.info("profile %s", profile)
    if args.sample is not None or args.sample_index is not None:
        LOG.info("wrote %s sample profile(s); rerun without --sample-index to merge", len(selected))
        return
    profiles = [out_dir / f"{sample.sample_id}.txt" for sample in samples]
    absent = [path for path in profiles if not path.exists()]
    if absent:
        raise MissingInputError(
            f"MetaPhlAn profiles for {name} are incomplete. "
            "Rerun without --sample-index after every sample profile exists.",
            absent,
        )
    merged = _merge(profiles, [sample.sample_id for sample in samples])
    destination.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(destination, sep="\t", index=False)
    LOG.info("wrote %s", destination)


def _resolve_manifest(cfg: Config, raw: Path) -> Path:
    if raw.is_file():
        return raw.resolve()
    candidate = cfg.repo_root / raw
    if candidate.is_file():
        return candidate.resolve()
    raise MissingInputError(f"sample manifest {raw}", [candidate])


def _cohort_for_manifest(cfg: Config, manifest: Path) -> Cohort | None:
    resolved = manifest.resolve()
    for cohort in cfg.cohorts.values():
        if cohort.manifest.resolve() == resolved:
            return cohort
    return None


def _seq_type(manifest: Path) -> str:
    frame = pd.read_csv(manifest, sep="\t", dtype=str)
    if "library_layout" not in frame.columns:
        return "illumina"
    layouts = {str(value).upper() for value in frame["library_layout"].dropna()}
    if layouts == {"SINGLE"}:
        return "pacbio"
    return "illumina"


def _select(samples: list[Sample], manifest: Path, args: argparse.Namespace) -> list[Sample]:
    chosen = samples
    if args.sample_index is not None:
        if args.sample_index < 0 or args.sample_index >= len(chosen):
            raise MissingInputError(
                f"sample index {args.sample_index} is outside manifest length {len(chosen)}",
                [manifest],
            )
        chosen = [chosen[args.sample_index]]
    if args.sample is not None:
        chosen = [sample for sample in chosen if sample.sample_id == args.sample]
        if not chosen:
            raise MissingInputError(f"sample {args.sample} is not in the manifest", [manifest])
    return chosen


def _missing_fastqs(samples: list[Sample], fastq_dir: Path, seq_type: str) -> list[Path]:
    missing: list[Path] = []
    for sample in samples:
        for path in fastq_paths(fastq_dir, sample.sample_id, seq_type):
            if not path.exists():
                missing.append(path)
    return missing


def _profile(cfg: Config, reads: list[Path], profile: Path) -> None:
    command = [
        "metaphlan",
        ",".join(str(path) for path in reads),
        "--input_type",
        "fastq",
        "--nproc",
        str(cfg.threads),
        "-o",
        str(profile),
    ]
    if cfg.metaphlan_db:
        command.extend(["--bowtie2db", cfg.metaphlan_db])
    run_command(command)


def _merge(profiles: list[Path], sample_ids: list[str]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for profile, sample_id in zip(profiles, sample_ids):
        frame = pd.read_csv(profile, sep="\t", comment="#")
        if "clade_name" not in frame.columns:
            raise MissingInputError("MetaPhlAn profile has no clade_name column", [profile])
        value_col = "relative_abundance" if "relative_abundance" in frame.columns else frame.columns[-1]
        piece = frame[["clade_name", value_col]].rename(columns={value_col: sample_id})
        merged = piece if merged is None else merged.merge(piece, on="clade_name", how="outer")
    if merged is None:
        raise MissingInputError("no MetaPhlAn profiles to merge", profiles)
    return merged.fillna(0)
