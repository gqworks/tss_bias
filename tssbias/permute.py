"""Position-permutation tests for coverage bias and GC bias.

The observed statistic is the mean of per-TSS biases. Each replicate rotates
TSS coordinates on every contig (see :func:`tssbias.stats.shift_tss_positions`),
draws a strand from that contig's + strand fraction, and recomputes the mean.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tssbias.coverage import bias_table, orient_windows
from tssbias.metrics import gc_bias_from_sequence
from tssbias.stats import greater_permutation_pvalue, rng_for, shift_tss_positions


@dataclass(frozen=True)
class PermutationResult:
    """One sample-by-assembly permutation test.

    Outputs of :func:`permute_coverage` and :func:`permute_gc`.
    """

    mean_statistic: float
    pvalue: float
    n_tss: int
    n_permutations: int
    null_mean: float


def _finite_mean(values: np.ndarray) -> tuple[float, int]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), 0
    return float(finite.mean()), int(finite.size)


def permute_coverage(
    depth: dict[str, np.ndarray],
    tss: pd.DataFrame,
    *,
    n_permutations: int,
    seed: int,
    sample: str,
    assembly: str,
    window_bp: int = 500,
    min_gene_bp: int = 0,
) -> PermutationResult:
    """Coverage-bias permutation test.

    Inputs:
        depth: contig coverage arrays.
        tss: filtered TSS table for one assembly.
        n_permutations: 500 for Illumina, 100 for long reads in the manuscript.
        seed: pipeline seed. Combined with sample and assembly.
        sample, assembly: labels mixed into the seed.
        window_bp: flank length.
        min_gene_bp: if positive, keep genes longer than this (long-read panel).
    Outputs:
        PermutationResult. ``mean_statistic`` is the mean coverage bias.
    """
    table = tss if min_gene_bp <= 0 else tss.loc[tss["gene_length"] > min_gene_bp].copy()
    windows, covered = orient_windows(depth, table, window_bp)
    observed_bias = bias_table(table, windows, covered, window_bp)["coverage_bias"].to_numpy()
    observed, n_tss = _finite_mean(observed_bias)
    rng = rng_for(seed, "coverage", sample, assembly, str(min_gene_bp))
    null = np.empty(n_permutations, dtype=float)
    for replicate in range(n_permutations):
        null[replicate] = _null_coverage_mean(depth, table, rng, window_bp)
    pvalue = greater_permutation_pvalue(observed, null)
    finite_null = null[np.isfinite(null)]
    null_mean = float(finite_null.mean()) if finite_null.size else float("nan")
    return PermutationResult(observed, pvalue, n_tss, int(finite_null.size), null_mean)


def _null_coverage_mean(
    depth: dict[str, np.ndarray],
    tss: pd.DataFrame,
    rng: np.random.Generator,
    window_bp: int,
) -> float:
    pieces: list[np.ndarray] = []
    for genome, subset in tss.groupby("genome", sort=False):
        array = depth.get(str(genome))
        if array is None or subset.empty:
            continue
        shifted = shift_tss_positions(subset["tss"].to_numpy(), rng)
        p_plus = float(np.mean(subset["strand"].to_numpy() == "+"))
        strands = np.where(rng.random(shifted.size) < p_plus, "+", "-")
        length = int(array.shape[0])
        keep = (shifted > window_bp) & (shifted + window_bp < length)
        if not np.any(keep):
            continue
        fake = pd.DataFrame(
            {
                "genome": str(genome),
                "tss": shifted[keep],
                "strand": strands[keep],
                "locus_tag": [f"null_{i}" for i in range(int(keep.sum()))],
            }
        )
        windows, covered = orient_windows(depth, fake, window_bp)
        bias = bias_table(fake, windows, covered, window_bp)["coverage_bias"].to_numpy()
        finite = bias[np.isfinite(bias)]
        if finite.size:
            pieces.append(finite)
    if not pieces:
        return float("nan")
    return float(np.concatenate(pieces).mean())


def permute_gc(
    sequences: dict[str, str],
    tss: pd.DataFrame,
    *,
    n_permutations: int,
    seed: int,
    sample: str,
    assembly: str,
    window_bp: int = 500,
) -> PermutationResult:
    """GC-bias permutation test. Same rotation as coverage.

    Inputs:
        sequences: contig sequences.
        tss: filtered TSS table.
        n_permutations, seed, sample, assembly, window_bp: as in
            :func:`permute_coverage`. ``sample`` is only part of the seed;
            GC does not depend on reads.
    Outputs:
        PermutationResult whose statistic is mean GC bias.
    """
    observed_values = _gc_values(sequences, tss, window_bp)
    observed, n_tss = _finite_mean(observed_values)
    rng = rng_for(seed, "gc", sample, assembly)
    null = np.empty(n_permutations, dtype=float)
    for replicate in range(n_permutations):
        null[replicate] = _null_gc_mean(sequences, tss, rng, window_bp)
    finite_null = null[np.isfinite(null)]
    null_mean = float(finite_null.mean()) if finite_null.size else float("nan")
    return PermutationResult(
        observed,
        greater_permutation_pvalue(observed, null),
        n_tss,
        int(finite_null.size),
        null_mean,
    )


def _gc_values(sequences: dict[str, str], tss: pd.DataFrame, window_bp: int) -> np.ndarray:
    values = []
    for row in tss.itertuples(index=False):
        sequence = sequences.get(str(row.genome))
        if sequence is None:
            values.append(np.nan)
            continue
        values.append(gc_bias_from_sequence(sequence, int(row.tss), str(row.strand), window_bp))
    return np.asarray(values, dtype=float)


def _null_gc_mean(
    sequences: dict[str, str],
    tss: pd.DataFrame,
    rng: np.random.Generator,
    window_bp: int,
) -> float:
    values: list[float] = []
    for genome, subset in tss.groupby("genome", sort=False):
        sequence = sequences.get(str(genome))
        if sequence is None or subset.empty:
            continue
        shifted = shift_tss_positions(subset["tss"].to_numpy(), rng)
        p_plus = float(np.mean(subset["strand"].to_numpy() == "+"))
        strands = np.where(rng.random(shifted.size) < p_plus, "+", "-")
        length = len(sequence)
        for position, strand in zip(shifted, strands):
            if position <= window_bp or position + window_bp >= length:
                continue
            values.append(gc_bias_from_sequence(sequence, int(position), str(strand), window_bp))
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())
