"""Command line. ``python -m tssbias <command> --cohort hmp2``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tssbias.commands import check_inputs, correlations, discover, download, expression, five_prime, gc_lmm
from tssbias.commands import mag, mappability, matched, metaphlan, mismatch, mock, permutation
from tssbias.commands import prepare_references, promotech, substitutions, sylph, upstream
from tssbias.commands.common import add_cohort_args
from tssbias.config import load_config
from tssbias.errors import MissingInputError, ToolError


def main(argv: list[str] | None = None) -> int:
    """Dispatch one analysis command.

    Inputs:
        argv: arguments without the program name. None reads sys.argv.
    Outputs:
        Process exit code. 0 on success, 2 when an input is missing, 3 when
        an external program fails.
    """
    parser = argparse.ArgumentParser(
        prog="tssbias",
        description="Re-execute the TSS coverage-bias manuscript analyses.",
    )
    parser.add_argument("--config", type=Path, default=None, help="YAML config. Default: config/default.yaml or TSS_CONFIG")
    sub = parser.add_subparsers(dest="command", required=True)

    def cohort_parser(name: str, help_text: str, func, extra=None):
        command = sub.add_parser(name, help=help_text)
        add_cohort_args(command)
        if extra:
            extra(command)
        command.set_defaults(func=func)
        return command

    cohort_parser("prepare-references", "Download RefSeq genomes and call Prodigal.", prepare_references.run)
    cohort_parser("discover", "Align reads and write ±500 bp TSS coverage matrices.", discover.run)
    cohort_parser("check-inputs", "Report missing files and tools for a cohort.", check_inputs.run)

    def perm_extra(command: argparse.ArgumentParser) -> None:
        command.add_argument("--metric", choices=("coverage", "gc"), default="coverage")
        command.add_argument("--min-gene-bp", type=int, default=0)
        command.add_argument("--fdr-only", action="store_true")

    cohort_parser("permutation", "Position-permutation test and BH FDR.", permutation.run, perm_extra)
    cohort_parser("expression", "Abundance-normalised TPM groups versus coverage bias.", expression.run)
    cohort_parser("gc-lmm", "GC-adjusted mixed model of expression.", gc_lmm.run)
    cohort_parser("mappability", "ART uniform reads and a permutation test.", mappability.run)
    cohort_parser("five-prime", "5' versus 3' read ends around the TSS.", five_prime.run)
    cohort_parser("mismatch-penalty", "Bowtie2 --mp sweep.", mismatch.run)
    cohort_parser("substitutions", "NM counts by expression group at --mp 2.", substitutions.run)
    cohort_parser("mag-bias", "MEGAHIT/MetaBAT2/Kraken bins versus RefSeq bias.", mag.run)
    cohort_parser("promotech", "Promotech promoters within 100 bp of Prodigal starts.", promotech.run)
    cohort_parser("sylph", "sylph adjusted ANI versus coverage bias.", sylph.run)
    cohort_parser("metaphlan", "Run MetaPhlAn and write the merged abundance table.", metaphlan.run)
    cohort_parser("upstream-scores", "Semi-global scores of upstream mates and the alignment LMM.", upstream.run)

    cohorts = sub.add_parser("cohort-correlation", help="Pearson correlation of bias across Illumina cohorts.")
    cohorts.set_defaults(func=correlations.run_cohorts)
    platforms = sub.add_parser("platform-comparison", help="Pearson correlation across Illumina, PacBio, and Nanopore.")
    platforms.set_defaults(func=correlations.run_platforms)
    mock_cmd = sub.add_parser("mock", help="ATCC MSA-1003 mock community coverage bias.")
    mock_cmd.add_argument(
        "--hifi",
        action="store_true",
        help="Contrast exact-strain bias with Illumina reads mapped to the HiFi reference.",
    )
    mock_cmd.set_defaults(func=mock.run)
    download_cmd = sub.add_parser(
        "download",
        help="Download SRA FASTQs named in a manifest with fasterq-dump.",
    )
    download_cmd.add_argument(
        "manifest",
        type=Path,
        help="Manifest TSV. FASTQs go to {data_root}/fastq/<filename stem>/",
    )
    download_cmd.set_defaults(func=download.run)
    matched_cmd = sub.add_parser("matched-longread", help="Short reads aligned to same-sample long reads.")
    matched_cmd.add_argument("--sample", default=None)
    matched_cmd.add_argument("--sample-index", type=int, default=None)
    matched_cmd.set_defaults(func=matched.run)

    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        args.func(cfg, args)
    except MissingInputError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except SystemExit as exc:
        code = exc.code
        return int(code) if isinstance(code, int) else (0 if code is None else 1)
    except ToolError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
