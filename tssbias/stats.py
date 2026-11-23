"""Position-permutation null and Benjamini–Hochberg FDR.

The null keeps the successive distances between TSS coordinates on a contig
and rotates them, which is the procedure in the original analysis and the
manuscript ("distances between them maintains the same distances"). The
modulus is the largest observed TSS coordinate on that contig.

The one-sided p-value is ``min(1, (#{null mean > observed mean} + 1) / n)``.
The lab scripts omitted the upper bound; values above 1 are clipped so the
FDR step receives a probability. Multiple testing is Benjamini–Hochberg at
the configured alpha (manuscript: 0.05) across sample-by-species tests.
"""

from __future__ import annotations

import hashlib

import numpy as np
from statsmodels.stats.multitest import multipletests


def rng_for(seed: int, *parts: str) -> np.random.Generator:
    """Deterministic generator keyed by the global seed and labels.

    Inputs:
        seed: pipeline seed from config.
        parts: sample id, assembly, and analysis name. Order is significant.
    Outputs:
        numpy Generator. The same inputs always rebuild the same stream.
    """
    mixes = [int(seed)]
    for part in parts:
        digest = hashlib.sha256(part.encode("utf-8")).digest()
        mixes.append(int.from_bytes(digest[:8], "little"))
    return np.random.default_rng(np.random.SeedSequence(mixes))


def shift_tss_positions(positions: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Rotate TSS coordinates while preserving successive gaps.

    Inputs:
        positions: 1-based TSS coordinates on one contig.
        rng: Generator used for the single rotation offset.
    Outputs:
        Shifted coordinates, same length. Modulus is max(positions), matching
        the original ``randomize`` helper. A single site is rotated the same way.
    """
    intervals = np.sort(np.asarray(positions, dtype=np.int64))
    if intervals.size == 0:
        return intervals
    max_len = int(intervals[-1])
    if max_len <= 0:
        return intervals.copy()
    shift = int(rng.integers(0, max_len))
    if intervals.size == 1:
        return np.array([(int(intervals[0]) + shift) % max_len], dtype=np.int64)
    gaps = intervals[1:] - intervals[:-1]
    new_points = np.concatenate([np.cumsum(gaps), [max_len]])
    return (new_points + shift) % max_len


def greater_permutation_pvalue(observed: float, null_means: np.ndarray) -> float:
    """One-sided permutation p-value for a mean that is larger than the null.

    Inputs:
        observed: mean coverage bias (or GC bias) at real TSS positions.
        null_means: mean bias at rotated positions, one value per replicate.
    Outputs:
        ``min(1, (count of null means strictly greater than observed + 1) / n)``.
        NaN if ``observed`` is not finite or no finite null means exist.
    """
    null = np.asarray(null_means, dtype=float)
    null = null[np.isfinite(null)]
    if null.size == 0 or not np.isfinite(observed):
        return float("nan")
    return float(min(1.0, (np.sum(null > observed) + 1) / null.size))


def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini–Hochberg adjusted p-values.

    Inputs:
        pvalues: one-sided permutation p-values, one per test.
        alpha: FDR level. The manuscript uses 0.05.
    Outputs:
        ``(reject, adjusted_p)``. Tests with non-finite p-values are not
        rejected and keep a NaN adjusted p-value. Adjustment is computed on
        the finite subset only, in the original order for those tests.
    """
    values = np.asarray(pvalues, dtype=float)
    reject = np.zeros(values.shape, dtype=bool)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return reject, adjusted
    rejected, adj, _, _ = multipletests(values[finite], alpha=alpha, method="fdr_bh")
    reject[finite] = rejected
    adjusted[finite] = adj
    return reject, adjusted
