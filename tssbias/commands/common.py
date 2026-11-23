"""Shared argument handling and pair selection for commands."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tssbias.cohorts import Sample, abundant_pairs, read_abundance, read_manifest, species_passing
from tssbias.config import Cohort, Config, Species
from tssbias.errors import MissingInputError
from tssbias.refseq import genomes_for


def add_cohort_args(parser: argparse.ArgumentParser) -> None:
    """Arguments shared by every per-cohort command."""
    parser.add_argument("--cohort", required=True, help="Cohort key in the config, e.g. hmp2")
    parser.add_argument("--sample", default=None, help="Run one manifest sample id")
    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="0-based index into the manifest (PBS array index minus 1)",
    )
    parser.add_argument("--species", default=None, help="Restrict to one s__ species name")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute outputs that already exist",
    )


def selected_samples(cfg: Config, cohort: Cohort, args: argparse.Namespace) -> list[Sample]:
    """Manifest rows after --sample / --sample-index.

    Inputs:
        cfg: unused except to keep the call symmetric; selection is by cohort.
        cohort: resolved cohort.
        args: namespace from :func:`add_cohort_args`.
    Outputs:
        Sample list. Raises MissingInputError if the index or id is absent.
    """
    del cfg
    samples = read_manifest(
        cohort.manifest,
        sample_column=cohort.sample_column,
        longread_column=cohort.longread_column or None,
    )
    if args.sample_index is not None:
        if args.sample_index < 0 or args.sample_index >= len(samples):
            raise MissingInputError(
                f"sample index {args.sample_index} is outside manifest length {len(samples)}",
                [cohort.manifest],
            )
        samples = [samples[args.sample_index]]
    if args.sample is not None:
        samples = [sample for sample in samples if sample.sample_id == args.sample]
        if not samples:
            raise MissingInputError(
                f"sample {args.sample} is not in the manifest",
                [cohort.manifest],
            )
    return samples


def screened_names(cfg: Config, cohort: Cohort, args: argparse.Namespace) -> list[str]:
    """MetaPhlAn species that pass this cohort's abundance screen.

    Mock is unscreened: the ATCC command does not use this list.
    """
    if cohort.name == "mock":
        raise MissingInputError(
            "cohort mock is not abundance-screened. "
            "Use `python -m tssbias mock` for the ATCC exact-strain reference.",
            [cohort.manifest],
        )
    table_path = abundance_path(cfg, cohort)
    abundance = read_abundance(table_path)
    samples = read_manifest(
        cohort.manifest,
        sample_column=cohort.sample_column,
        longread_column=cohort.longread_column or None,
    )
    names = species_passing(
        abundance,
        [sample.sample_id for sample in samples],
        cohort.abundance_min_pct,
        cohort.abundance_min_samples,
    )
    if args.species is not None:
        if args.species not in names:
            raise MissingInputError(
                f"{args.species} did not pass the {cohort.abundance_min_pct}% / "
                f"{cohort.abundance_min_samples}-sample screen for cohort {cohort.name}",
                [table_path],
            )
        names = [args.species]
    if not names:
        raise MissingInputError(
            f"no species passed the {cohort.abundance_min_pct}% / "
            f"{cohort.abundance_min_samples}-sample screen for cohort {cohort.name}",
            [table_path, cohort.manifest],
        )
    return names


def selected_species(cfg: Config, cohort: Cohort, args: argparse.Namespace) -> list[Species]:
    """Screened species whose RefSeq genome prepare-references has recorded."""
    return genomes_for(cfg.refseq_dir, screened_names(cfg, cohort, args))


def abundance_path(cfg: Config, cohort: Cohort) -> Path:
    """Abundance table for a cohort."""
    return cohort.abundance_table or cfg.mpa_merged


def passing_pairs(
    cfg: Config,
    cohort: Cohort,
    args: argparse.Namespace,
) -> list[tuple[Sample, Species]]:
    """Sample-species pairs that pass the abundance screen.

    Inputs:
        cfg, cohort, args: command context.
    Outputs:
        Pairs whose relative abundance is at least the cohort minimum.
        Raises MissingInputError when the table or manifest samples are absent.
    """
    samples = selected_samples(cfg, cohort, args)
    species = selected_species(cfg, cohort, args)
    table_path = abundance_path(cfg, cohort)
    # Screen on the full manifest so the "at least N samples" rule is the
    # cohort rule, not the one-task subset from a job array.
    full_manifest = read_manifest(
        cohort.manifest,
        sample_column=cohort.sample_column,
        longread_column=cohort.longread_column or None,
    )
    abundance = read_abundance(table_path)
    mask = abundant_pairs(
        abundance,
        full_manifest,
        [item.name for item in species],
        cohort.abundance_min_pct,
        cohort.abundance_min_samples,
    )
    wanted = {sample.sample_id for sample in samples}
    pairs: list[tuple[Sample, Species]] = []
    by_id = {sample.sample_id: sample for sample in samples}
    name_to_species = {item.name: item for item in species}
    for species_name in mask.index:
        for sample_id in mask.columns:
            if sample_id not in wanted:
                continue
            if bool(mask.at[species_name, sample_id]):
                pairs.append((by_id[sample_id], name_to_species[species_name]))
    if not pairs:
        raise MissingInputError(
            f"no sample-species pairs passed the {cohort.abundance_min_pct}% / "
            f"{cohort.abundance_min_samples}-sample screen for cohort {cohort.name}",
            [table_path, cohort.manifest],
        )
    return pairs


def write_table(path: Path, frame: pd.DataFrame) -> None:
    """Write a CSV, creating parent directories.

    Inputs:
        path: destination.
        frame: table.
    Outputs:
        File at ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
