#!/usr/bin/env python3
"""Generate RefSeq genomes, alignments, and ±500 bp TSS coverage matrices.

Discovery cohorts (Franzosa / HMP2 / ARG / alt Illumina) are enabled via CLI
flags — not triple-quoted notebook blocks. Paths come from config YAML.

Examples:
  python calc_coverage.py --list-cohorts
  python calc_coverage.py --cohort hmp2 --dry-run
  python calc_coverage.py --cohort hmp2 --cohort franzosa
  python calc_coverage.py --cohort longread
  python calc_coverage.py --all-illumina
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tss_bias.config import ensure_workdir_dirs, load_config


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="YAML config (default: config/default.yaml or $TSS_CONFIG)")
    p.add_argument(
        "--cohort",
        action="append",
        dest="cohorts",
        choices=["hmp2", "franzosa", "arg", "hyp", "alt_illumina", "longread", "pacbio", "nanopore"],
        help="Cohort(s) to process (repeatable). alt_illumina = franzosa+hmp2+arg+hyp intersection filter.",
    )
    p.add_argument("--all-illumina", action="store_true", help="Run franzosa, hmp2, and alt_illumina stages")
    p.add_argument("--list-cohorts", action="store_true", help="Print configured cohorts and exit")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing aligners")
    p.add_argument(
        "--skip-mpa-check",
        action="store_true",
        help="Allow missing MetaPhlAn table (dry planning only; still errors on real run without --dry-run)",
    )
    return p


def _run_franzosa(cfg, threads, dry_run):
    import numpy as np

    from tss_bias.coverage import run_cohort_coverage
    from tss_bias.io import filter_species_abundance, load_mpa_species_table, load_sample_to_fastq_dir
    from tss_bias.species import ftp_list_for_cohort

    samples = load_sample_to_fastq_dir(cfg, "franzosa")
    mpa = load_mpa_species_table(cfg)
    c = cfg["cohorts"]["franzosa"]
    abund = filter_species_abundance(
        mpa, list(samples), c["abundance_min_pct"], c["abundance_min_samples"]
    )
    ftps = ftp_list_for_cohort(c["species_set"])
    run_cohort_coverage(cfg, "franzosa", samples, abund, ftps, "illumina", threads, dry_run)


def _run_hmp2(cfg, threads, dry_run):
    from tss_bias.coverage import run_cohort_coverage
    from tss_bias.io import filter_species_abundance, load_mpa_species_table, load_sample_to_fastq_dir
    from tss_bias.species import ftp_list_for_cohort

    samples = load_sample_to_fastq_dir(cfg, "hmp2")
    mpa = load_mpa_species_table(cfg)
    c = cfg["cohorts"]["hmp2"]
    abund = filter_species_abundance(
        mpa, list(samples), c["abundance_min_pct"], c["abundance_min_samples"]
    )
    ftps = ftp_list_for_cohort(c["species_set"])
    run_cohort_coverage(cfg, "hmp2", samples, abund, ftps, "illumina", threads, dry_run)


def _run_arg(cfg, threads, dry_run):
    from tss_bias.coverage import run_cohort_coverage
    from tss_bias.io import filter_species_abundance, load_mpa_species_table, load_sample_to_fastq_dir
    from tss_bias.species import ftp_list_for_cohort

    samples = load_sample_to_fastq_dir(cfg, "arg")
    mpa = load_mpa_species_table(cfg)
    c = cfg["cohorts"]["arg"]
    abund = filter_species_abundance(
        mpa, list(samples), c["abundance_min_pct"], c["abundance_min_samples"]
    )
    ftps = ftp_list_for_cohort(c["species_set"])
    run_cohort_coverage(cfg, "arg", samples, abund, ftps, "illumina", threads, dry_run)


def _run_alt_illumina(cfg, threads, dry_run):
    """Species present above threshold in Franzosa AND HMP2 AND ARG AND HYP."""
    import numpy as np

    from tss_bias.coverage import run_cohort_coverage
    from tss_bias.io import load_mpa_species_table, load_sample_to_fastq_dir
    from tss_bias.species import ftp_list_for_cohort

    franzosa = load_sample_to_fastq_dir(cfg, "franzosa")
    hmp2 = load_sample_to_fastq_dir(cfg, "hmp2")
    arg = load_sample_to_fastq_dir(cfg, "arg")
    hyp = load_sample_to_fastq_dir(cfg, "hyp")
    merged_samples = {**franzosa, **hmp2, **arg, **hyp}
    mpa = load_mpa_species_table(cfg)
    cols = [c for c in merged_samples if c in mpa.columns]
    sub = mpa.loc[:, cols]
    fr_cols = [c for c in franzosa if c in sub.columns]
    hm_cols = [c for c in hmp2 if c in sub.columns]
    ar_cols = [c for c in arg if c in sub.columns]
    hy_cols = [c for c in hyp if c in sub.columns]
    mask = (
        ((sub[fr_cols] > 0.5).sum(axis=1) >= 5)
        & ((sub[hm_cols] > 0.5).sum(axis=1) >= 30)
        & ((sub[ar_cols] > 0.5).sum(axis=1) >= 5)
        & ((sub[hy_cols] > 0.5).sum(axis=1) >= 5)
    )
    abund = sub.loc[mask, :].where(sub.loc[mask, :] > 0.5, np.nan)
    ftps = ftp_list_for_cohort("alt_illumina")
    run_cohort_coverage(cfg, "alt_illumina", merged_samples, abund, ftps, "illumina", threads, dry_run)


def _run_longread(cfg, threads, dry_run):
    import pandas as pd

    from tss_bias.coverage import run_cohort_coverage
    from tss_bias.io import (
        filter_species_abundance,
        load_longread_abundance,
        load_mpa_species_table,
        load_sample_to_fastq_dir,
    )
    from tss_bias.species import ftp_list_for_cohort

    hmp2 = load_sample_to_fastq_dir(cfg, "hmp2")
    pacbio = load_sample_to_fastq_dir(cfg, "pacbio")
    nanopore = load_sample_to_fastq_dir(cfg, "nanopore")
    mpa = load_mpa_species_table(cfg)
    hmp2_abund = filter_species_abundance(mpa, list(hmp2), 0.5, 30)
    pacbio_abund = load_longread_abundance(cfg, "pacbio", list(pacbio))
    nanopore_abund = load_longread_abundance(cfg, "nanopore", list(nanopore))
    merged_abund = pd.concat([hmp2_abund, pacbio_abund, nanopore_abund], axis=1)
    merged_abund = merged_abund.loc[
        ((merged_abund[list(hmp2)] > 0.5).sum(axis=1) >= 30)
        & ((merged_abund[list(pacbio)] > 0.5).sum(axis=1) >= 4)
        & ((merged_abund[list(nanopore)] > 0.5).sum(axis=1) >= 4),
        :,
    ]
    merged_abund = merged_abund.where(merged_abund > 0.5, np.nan)
    merged_samples = {**hmp2, **pacbio, **nanopore}
    seq_types = {
        **{s: "illumina" for s in hmp2},
        **{s: "pacbio" for s in pacbio},
        **{s: "nanopore" for s in nanopore},
    }
    ftps = ftp_list_for_cohort("longread")
    run_cohort_coverage(
        cfg, "longread", merged_samples, merged_abund, ftps, "illumina", threads, dry_run, seq_types
    )


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = load_config(args.config)
    ensure_workdir_dirs(cfg)

    if args.list_cohorts:
        for name, meta in cfg["cohorts"].items():
            print(f"{name:12} sample_list={meta['sample_list']}  fastq_dir={meta['fastq_dir']}")
        print(f"\nmpa_merged={cfg['paths']['mpa_merged']}")
        print(f"config={cfg['_config_path']}")
        return 0

    cohorts = list(args.cohorts or [])
    if args.all_illumina:
        cohorts.extend(["franzosa", "hmp2", "alt_illumina"])
    # Deduplicate preserving order
    seen = set()
    cohorts = [c for c in cohorts if not (c in seen or seen.add(c))]

    if not cohorts:
        print(
            "No cohort selected. A naive run no longer silently executes only long-read.\n"
            "Pass --cohort hmp2 (discovery) and/or --all-illumina, or --list-cohorts.\n"
            "See README.md / docs/GADI.md.",
            file=sys.stderr,
        )
        return 2

    threads = args.threads or cfg.get("compute", {}).get("threads")

    from tss_bias.coverage import run_cohort_coverage
    from tss_bias.io import (
        filter_species_abundance,
        load_mpa_species_table,
        load_sample_to_fastq_dir,
    )
    from tss_bias.species import ftp_list_for_cohort

    # Fail early on missing MPA unless dry-run + skip
    mpa_missing = False
    if any(c in cohorts for c in ("hmp2", "franzosa", "arg", "hyp", "alt_illumina", "longread")):
        try:
            load_mpa_species_table(cfg)
        except FileNotFoundError as e:
            if args.dry_run and args.skip_mpa_check:
                print(f"WARNING: {e}")
                mpa_missing = True
            else:
                print(e, file=sys.stderr)
                return 1

    if mpa_missing:
        for c in cohorts:
            samples = (
                load_sample_to_fastq_dir(cfg, c)
                if c in cfg["cohorts"]
                else {}
            )
            print(
                f"[dry-run] cohort={c} samples={len(samples)} "
                f"fastq_dir={cfg['cohorts'].get(c, {}).get('fastq_dir', 'n/a')} "
                f"(skipped align loop: no MetaPhlAn table)"
            )
        return 0

    dispatch = {
        "franzosa": _run_franzosa,
        "hmp2": _run_hmp2,
        "arg": _run_arg,
        "alt_illumina": _run_alt_illumina,
        "longread": _run_longread,
    }
    for c in cohorts:
        if c == "hyp":
            print("Note: hyp is included via --cohort alt_illumina; standalone hyp uses same pinned genomes.")
            samples = load_sample_to_fastq_dir(cfg, "hyp")
            mpa = load_mpa_species_table(cfg)
            meta = cfg["cohorts"]["hyp"]
            abund = filter_species_abundance(
                mpa, list(samples), meta["abundance_min_pct"], meta["abundance_min_samples"]
            )
            ftps = ftp_list_for_cohort(meta["species_set"])
            run_cohort_coverage(cfg, "hyp", samples, abund, ftps, "illumina", threads, args.dry_run)
            continue
        if c in ("pacbio", "nanopore"):
            print(f"Use --cohort longread for platform-aware PacBio/ONT+HMP2 joint filter (requested {c}).")
            continue
        print(f"=== Running cohort: {c} ===")
        dispatch[c](cfg, threads, args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
