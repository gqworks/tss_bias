#!/usr/bin/env python3
"""GC-adjusted association of TSS coverage bias with expression (LMM stub).

STATUS: The revision manuscript claims a linear mixed model relating coverage
bias to expression after adjusting for GC bias. **No MixedLM (or equivalent)
implementation existed in this repository.** This script is an explicit stub so
the gap is visible in the Gadi launch path — it does not invent results.

If/when the original analysis notebook or stats code is recovered, replace the
`NotImplementedError` path with the real model (suggested formula below).

Suggested model (for implementation — not run here):
    bias ~ expression_group + gc_bias + (1|species) + (1|sample)
using statsmodels.formula.api.mixedlm on a long table of gene- or
species–sample–level bias, GC (down−up), and abundance-normalized TPM group.

Usage:
  python scripts/gc_adjusted_lmm.py --help
  python scripts/gc_adjusted_lmm.py --input path/to/bias_gc_tpm_table.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


STUB_MESSAGE = """
GC-adjusted LMM is not implemented in this repository.

Manuscript claim: coverage bias remains associated with expression after
adjusting for GC bias at the TSS.

What exists today:
  - GC% down−up and expression-stratified boxplots: plot_bias_vs_tpm.py
  - GC null permutations: plot_heatmap_scatter.py --stage gc
  - No statsmodels MixedLM / lme4 / equivalent fit in any tracked .py file

Next steps for Gordon:
  1. Locate the original LMM notebook/script if it lived outside git, OR
  2. Re-fit from exported long tables (bias, gc_bias, tpm_group, species, sample), OR
  3. Soften/remove the LMM claim in the manuscript if unreproducible.
""".strip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--input",
        type=Path,
        help="Optional long table CSV (unused until model is ported)",
    )
    ap.add_argument(
        "--write-schema",
        type=Path,
        help="Write an empty schema CSV describing expected columns, then exit 0",
    )
    args = ap.parse_args(argv)

    if args.write_schema:
        args.write_schema.parent.mkdir(parents=True, exist_ok=True)
        args.write_schema.write_text(
            "species,sample,gene_id,bias,gc_bias,tpm_group,norm_log_tpm\n"
        )
        print(f"Wrote schema stub: {args.write_schema}")
        return 0

    print(STUB_MESSAGE, file=sys.stderr)
    if args.input:
        print(f"(Received --input {args.input} but refusing to invent an LMM fit.)", file=sys.stderr)
    raise SystemExit(
        "NotImplementedError: GC-adjusted LMM missing; see scripts/gc_adjusted_lmm.py docstring."
    )


if __name__ == "__main__":
    raise SystemExit(main())
