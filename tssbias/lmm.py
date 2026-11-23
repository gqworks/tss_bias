"""REML linear mixed models from the revised manuscript.

GC-adjusted expression model (Methods, LmerTest in the manuscript)::

    scale(log(norm_tpm + 0.001)) ~ scale(coverage_bias) + scale(gc_bias)
        + (1 | sample) + (1 | species)

``norm_tpm = 1e6 * kallisto_tpm / mapped_metagenomic_reads``. The log is
natural; standard-scaling afterwards makes the log base irrelevant.
``scale`` is a z-score with sample standard deviation (ddof=1) computed on
the rows that enter the model. TSSs whose 1001 bp window has no coverage are
excluded before scaling.

Alignment-score model (figure legend)::

    scale(coverage_bias) ~ scale(upstream_alignment_score)
        + (1 | sample) + (1 | species)

Both are fit by REML with crossed random intercepts. The manuscript used
lmerTest, which reports Satterthwaite degrees of freedom. This Python rerun
uses statsmodels ``MixedLM`` and reports Wald z tests. The fixed-effect
formula and the random effects are the same.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from tssbias.metrics import zscore


@dataclass(frozen=True)
class LmmResult:
    """Fitted mixed model.

    Inputs are the columns described on the class. Outputs of ``fit_crossed_lmm``
    populate every field.
    """

    formula: str
    n_obs: int
    n_samples: int
    n_species: int
    converged: bool
    params: dict[str, float]
    bse: dict[str, float]
    pvalues: dict[str, float]
    summary: str


def _scale_frame(frame: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    scaled = frame.copy()
    for source, dest in columns.items():
        scaled[dest] = zscore(scaled[source].to_numpy(dtype=float))
    return scaled


def fit_crossed_lmm(
    frame: pd.DataFrame,
    formula: str,
    scale_columns: dict[str, str],
    sample_col: str = "sample",
    species_col: str = "species",
) -> LmmResult:
    """Fit a REML mixed model with sample and species random intercepts.

    Inputs:
        frame: one row per TSS that already passed the coverage filter.
            Must contain ``sample_col``, ``species_col``, and every source
            column in ``scale_columns``.
        formula: patsy formula written in the *scaled* column names.
        scale_columns: map of raw column -> scaled column. Each raw column
            is z-scored across ``frame``.
        sample_col: metagenomic sample id.
        species_col: microbial species id.
    Outputs:
        LmmResult. Raises ValueError if fewer than two samples or two species
        remain, or if a column has zero variance. Raises RuntimeError if the
        optimiser does not converge.
    """
    needed = [sample_col, species_col, *scale_columns]
    missing = [column for column in needed if column not in frame.columns]
    if missing:
        raise ValueError(f"mixed model frame is missing columns: {missing}")
    data = frame.dropna(subset=needed).copy()
    n_samples = int(data[sample_col].nunique())
    n_species = int(data[species_col].nunique())
    if n_samples < 2 or n_species < 2:
        raise ValueError(
            "mixed model needs at least 2 samples and 2 species, "
            f"found {n_samples} samples and {n_species} species"
        )
    data = _scale_frame(data, scale_columns)
    # Species is the grouping factor (few levels). Sample is a crossed
    # variance component. Both are random intercepts, matching (1|sample)+(1|species).
    model = smf.mixedlm(
        formula,
        data,
        groups=data[species_col],
        re_formula="1",
        vc_formula={"sample": "0 + C(sample)"},
    )
    fitted = model.fit(method="lbfgs", reml=True, maxiter=500, disp=False)
    if not bool(fitted.converged):
        raise RuntimeError("mixed model did not converge; no coefficients were written")
    names = list(fitted.fe_params.index)
    params = {name: float(fitted.fe_params[name]) for name in names}
    bse = {name: float(fitted.bse_fe[name]) for name in names}
    pvalues = {name: float(fitted.pvalues[name]) for name in names}
    header = (
        f"Formula: {formula}\n"
        "Random effects: (1 | sample) + (1 | species)\n"
        "Estimator: REML, statsmodels MixedLM, Wald z p-values.\n"
        "Manuscript software was lmerTest (Satterthwaite df); the fixed effects "
        "and random effects match that model.\n"
        f"N={len(data)}  samples={n_samples}  species={n_species}\n"
    )
    return LmmResult(
        formula=formula,
        n_obs=int(len(data)),
        n_samples=n_samples,
        n_species=n_species,
        converged=True,
        params=params,
        bse=bse,
        pvalues=pvalues,
        summary=header + fitted.summary().as_text(),
    )


def prepare_gc_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop TSSs with no coverage and require finite bias, GC, and TPM.

    Inputs:
        frame: gene table with ``norm_tpm``, ``coverage_bias``, ``gc_bias``,
            ``covered`` (bool, False when the 1001 bp sum is 0), ``sample``,
            ``species``.
    Outputs:
        Copy ready for ``fit_crossed_lmm``. ``log_norm_tpm`` is added when absent.
    """
    data = frame.copy()
    if "covered" in data.columns:
        data = data.loc[data["covered"].astype(bool)]
    if "log_norm_tpm" not in data.columns:
        data["log_norm_tpm"] = np.log(data["norm_tpm"].to_numpy(dtype=float) + 0.001)
    keep = ["sample", "species", "log_norm_tpm", "coverage_bias", "gc_bias"]
    return data.dropna(subset=keep)
