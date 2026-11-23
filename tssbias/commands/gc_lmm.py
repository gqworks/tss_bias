"""GC-adjusted linear mixed model of expression.

Formula (manuscript Methods; natural log, then standard scaling)::

    scale(log(norm_tpm + 0.001)) ~ scale(coverage_bias) + scale(gc_bias)
        + (1 | sample) + (1 | species)

``norm_tpm = 1e6 * TPM / mapped_metagenomic_reads``. Rows with no coverage in
the 1001 bp window are excluded. Fit is REML. Wald z p-values are reported
because statsmodels does not implement the lmerTest Satterthwaite correction.
"""

from __future__ import annotations

import argparse

import pandas as pd

from tssbias.commands.common import write_table
from tssbias.config import Config
from tssbias.errors import MissingInputError
from tssbias.lmm import fit_crossed_lmm, prepare_gc_frame
from tssbias.logutil import get_logger
from tssbias.metrics import GC_LMM_FORMULA

LOG = get_logger("tssbias.gc-lmm")


def run(cfg: Config, args: argparse.Namespace) -> None:
    """Fit the GC-adjusted expression model for one cohort.

    Inputs:
        cfg, args: ``--cohort``. Reads the gene table from ``expression``.
    Outputs:
        ``results/lmm/{cohort}_gc_adjusted_lmm.txt`` with the formula and fit.
        ``results/lmm/{cohort}_gc_adjusted_coefficients.csv``.
    """
    cohort = cfg.cohort(args.cohort)
    table_path = cfg.results_dir / "expression" / f"{cohort.name}_gene_table.csv"
    if not table_path.exists():
        raise MissingInputError(
            "GC mixed model needs the expression gene table. "
            f"Run: python -m tssbias expression --cohort {cohort.name}",
            [table_path],
        )
    genes = pd.read_csv(table_path)
    if "gc_bias" not in genes.columns or genes["gc_bias"].notna().sum() == 0:
        raise MissingInputError(
            "gene table has no GC bias. Reference FASTAs were absent when expression ran. "
            f"Run prepare-references, then expression, for cohort {cohort.name}.",
            [cfg.refseq_dir, table_path],
        )
    model_frame = prepare_gc_frame(genes)
    result = fit_crossed_lmm(
        model_frame,
        formula="expr_z ~ bias_z + gc_z",
        scale_columns={
            "log_norm_tpm": "expr_z",
            "coverage_bias": "bias_z",
            "gc_bias": "gc_z",
        },
    )
    out_dir = cfg.results_dir / "lmm"
    out_dir.mkdir(parents=True, exist_ok=True)
    text_path = out_dir / f"{cohort.name}_gc_adjusted_lmm.txt"
    text_path.write_text(
        "Manuscript formula:\n"
        f"  {GC_LMM_FORMULA}\n"
        "norm_tpm = 1e6 * kallisto_tpm / mapped_metagenomic_reads\n"
        "log is the natural log. scale is a z-score (sample sd, ddof=1) on the rows used here.\n"
        "TSSs with zero coverage across the 1001 bp window are excluded.\n\n"
        + result.summary
        + "\n"
    )
    coefficients = pd.DataFrame(
        {
            "term": list(result.params),
            "estimate": [result.params[name] for name in result.params],
            "std_error": [result.bse[name] for name in result.params],
            "wald_pvalue": [result.pvalues[name] for name in result.params],
        }
    )
    write_table(out_dir / f"{cohort.name}_gc_adjusted_coefficients.csv", coefficients)
    LOG.info("wrote %s", text_path)
