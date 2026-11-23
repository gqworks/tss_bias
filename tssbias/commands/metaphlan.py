"""Run MetaPhlAn and merge species profiles.

The manuscript text says MetaPhlAn3. The tables the revision analyses read are
a merged species profile at ``mpa4_res/merged_mpa.txt``. This command runs
whatever ``metaphlan`` is on PATH (pin MetaPhlAn 4 in environment.yml for a
Gadi rerun) and writes that merged file. It does not relabel a MetaPhlAn3
fragment as MetaPhlAn4.
"""

from __future__ import annotations

import argparse

import pandas as pd

from tssbias.cohorts import fastq_paths
from tssbias.commands.common import abundance_path, selected_samples
from tssbias.config import Config
from tssbias.errors import MissingInputError
from tssbias.external import require_tool, run_command
from tssbias.logutil import get_logger

LOG = get_logger("tssbias.metaphlan")


def run(cfg: Config, args: argparse.Namespace) -> None:
    """Profile each sample and write the cohort abundance table.

    Outputs:
        Per-sample profiles under ``{data_root}/mpa4_res/{cohort}/``.
        The merged table at the cohort's ``abundance_table`` path.
    """
    cohort = cfg.cohort(args.cohort)
    samples = selected_samples(cfg, cohort, args)
    require_tool("metaphlan")
    out_dir = cfg.data_root / "mpa4_res" / cohort.name
    out_dir.mkdir(parents=True, exist_ok=True)
    profiles = []
    for sample in samples:
        reads = fastq_paths(cohort.fastq_dir, sample.sample_id, cohort.seq_type)
        missing = [path for path in reads if not path.exists()]
        if missing:
            raise MissingInputError(f"FASTQs for MetaPhlAn sample {sample.sample_id}", missing)
        profile = out_dir / f"{sample.sample_id}.txt"
        if args.force or not profile.exists():
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
        profiles.append(profile)
    if args.sample is not None or args.sample_index is not None:
        LOG.info("wrote %s sample profiles; rerun without --sample-index to merge", len(profiles))
        return
    profiles = sorted(out_dir.glob("*.txt"))
    sample_ids = [path.stem for path in profiles]
    merged = _merge(profiles, sample_ids)
    destination = abundance_path(cfg, cohort)
    destination.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(destination, sep="\t")
    LOG.info("wrote %s", destination)


def _merge(profiles: list, sample_ids: list[str]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for profile, sample_id in zip(profiles, sample_ids):
        frame = pd.read_csv(profile, sep="\t", comment="#")
        if "clade_name" not in frame.columns:
            raise MissingInputError("MetaPhlAn profile has no clade_name column", [profile])
        value_col = "relative_abundance" if "relative_abundance" in frame.columns else frame.columns[-1]
        piece = frame[["clade_name", value_col]].rename(columns={value_col: sample_id})
        merged = piece if merged is None else merged.merge(piece, on="clade_name", how="outer")
    assert merged is not None
    return merged.fillna(0)
