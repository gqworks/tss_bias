"""TSS coverage bias, GC bias, and abundance-normalised expression.

Definitions follow the revised manuscript, not the lab scripts' sum of an
asymmetric window.

* TSS is the Prodigal gene start (GFF start on ``+``, GFF end on ``-``).
* A window holds 1001 coverage values: 500 bp upstream, the TSS base, and
  500 bp downstream. Minus-strand windows are reversed so downstream is to
  the right.
* Each window is standard-scaled with the sample standard deviation
  (ddof=1), the same estimator as ``pandas.DataFrame.std``.
* Coverage bias is the mean of the scaled downstream flank minus the mean
  of the scaled upstream flank. The TSS base is in the scaled window and in
  neither mean.
* GC bias is GC% in the 500 bp downstream minus GC% in the 500 bp upstream.
* Abundance-normalised TPM is ``1e6 * kallisto_tpm / mapped_metagenomic_reads``.
* Expression groups use that normalised TPM: zero (==0), low (<100),
  mid (<1000), high (>=1000). The log used downstream adds pseudocount 0.001
  and is the natural log.
"""

from __future__ import annotations

import numpy as np

GC_LMM_FORMULA = (
    "scale(log(norm_tpm + 0.001)) ~ scale(coverage_bias) + scale(gc_bias)"
    " + (1 | sample) + (1 | species)"
)
ALIGNMENT_LMM_FORMULA = (
    "scale(coverage_bias) ~ scale(upstream_alignment_score)"
    " + (1 | sample) + (1 | species)"
)
EXPRESSION_GROUPS = ("zero", "low", "mid", "high")


def window_length(window_bp: int = 500) -> int:
    """Return the number of bases in a TSS window, including the TSS base."""
    return 2 * window_bp + 1


def scale_rows(windows: np.ndarray) -> np.ndarray:
    """Standard-scale each window independently.

    Inputs:
        windows: array of shape (n, 2 * window_bp + 1).
    Outputs:
        Array of the same shape. Rows with zero sample variance are NaN.
    """
    values = np.asarray(windows, dtype=float)
    mean = values.mean(axis=1)
    std = values.std(axis=1, ddof=1)
    scaled = np.full(values.shape, np.nan, dtype=float)
    ok = np.isfinite(std) & (std > 0)
    if np.any(ok):
        scaled[ok] = (values[ok] - mean[ok, None]) / std[ok, None]
    return scaled


def coverage_bias(windows: np.ndarray, window_bp: int = 500) -> np.ndarray:
    """Mean scaled downstream coverage minus mean scaled upstream coverage.

    Inputs:
        windows: coverage of shape (n, 2 * window_bp + 1), already oriented
            so column ``window_bp`` is the TSS and higher columns are downstream.
        window_bp: flank length. The manuscript uses 500.
    Outputs:
        Vector of length n. NaN where the window has undefined scale (no
        coverage, or constant coverage).
    """
    values = np.asarray(windows, dtype=float)
    expected = window_length(window_bp)
    if values.ndim != 2 or values.shape[1] != expected:
        raise ValueError(f"windows must have shape (n, {expected}), got {values.shape}")
    scaled = scale_rows(values)
    upstream = _row_mean(scaled[:, :window_bp])
    downstream = _row_mean(scaled[:, window_bp + 1 :])
    return downstream - upstream


def _row_mean(values: np.ndarray) -> np.ndarray:
    counts = np.sum(np.isfinite(values), axis=1)
    sums = np.nansum(values, axis=1)
    out = np.full(values.shape[0], np.nan, dtype=float)
    ok = counts > 0
    out[ok] = sums[ok] / counts[ok]
    return out


def gc_percent(sequence: str) -> float:
    """GC percent of a sequence. Ambiguous bases count in the denominator.

    Inputs:
        sequence: nucleotide string.
    Outputs:
        100 * (G+C) / length. NaN for an empty string.
    """
    text = sequence.upper()
    if not text:
        return float("nan")
    gc = text.count("G") + text.count("C")
    return 100.0 * gc / len(text)


def gc_bias_from_sequence(sequence: str, tss_1based: int, strand: str, window_bp: int = 500) -> float:
    """GC% downstream minus GC% upstream of one TSS.

    Inputs:
        sequence: contig sequence in reference orientation.
        tss_1based: Prodigal gene start, 1-based inclusive.
        strand: ``+`` or ``-``. Downstream is into the gene.
        window_bp: flank length (500 in the manuscript).
    Outputs:
        Scalar GC bias. NaN if either flank runs off the contig.
    """
    t0 = int(tss_1based) - 1
    if strand == "+":
        up_start, up_end = t0 - window_bp, t0
        down_start, down_end = t0 + 1, t0 + 1 + window_bp
    elif strand == "-":
        up_start, up_end = t0 + 1, t0 + 1 + window_bp
        down_start, down_end = t0 - window_bp, t0
    else:
        raise ValueError(f"strand must be + or -, got {strand!r}")
    if min(up_start, down_start) < 0 or max(up_end, down_end) > len(sequence):
        return float("nan")
    return gc_percent(sequence[down_start:down_end]) - gc_percent(sequence[up_start:up_end])


def abundance_normalised_tpm(
    tpm: np.ndarray,
    mapped_reads: float,
    pseudocount: float = 0.001,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalise kallisto TPM by metagenomic reads mapped to the reference.

    Inputs:
        tpm: kallisto TPM values.
        mapped_reads: total reads aligned to that reference in the matched
            metagenome (samtools idxstats sum). Must be positive.
        pseudocount: added after normalisation, before the log. Manuscript
            value is 0.001.
    Outputs:
        ``(norm_tpm, log_norm_tpm)`` where
        ``norm_tpm = 1e6 * tpm / mapped_reads`` and
        ``log_norm_tpm = ln(norm_tpm + pseudocount)``.
    """
    if mapped_reads <= 0:
        raise ValueError("mapped_reads must be positive to normalise TPM")
    values = np.asarray(tpm, dtype=float)
    norm = 1e6 * values / float(mapped_reads)
    return norm, np.log(norm + pseudocount)


def expression_group(norm_tpm: np.ndarray, low: float = 100.0, mid: float = 1000.0) -> np.ndarray:
    """Bin abundance-normalised TPM into zero / low / mid / high.

    Inputs:
        norm_tpm: abundance-normalised TPM (not the log).
        low: upper bound of the low bin (exclusive). Manuscript: 100.
        mid: upper bound of the mid bin (exclusive). Manuscript: 1000.
    Outputs:
        Object array of labels. Non-finite inputs are None.
        Boundaries: ==0 zero; 0 < x < low low; low <= x < mid mid; x >= mid high.
    """
    values = np.asarray(norm_tpm, dtype=float)
    labels = np.empty(values.shape, dtype=object)
    labels[:] = "high"
    labels[values < mid] = "mid"
    labels[values < low] = "low"
    labels[values == 0] = "zero"
    labels[~np.isfinite(values)] = None
    return labels


def zscore(values: np.ndarray) -> np.ndarray:
    """Standard-scale a 1-d vector with sample standard deviation.

    Inputs:
        values: finite observations.
    Outputs:
        (values - mean) / std(ddof=1). Raises ValueError if std is 0.
    """
    array = np.asarray(values, dtype=float)
    std = array.std(ddof=1)
    if not np.isfinite(std) or std == 0:
        raise ValueError("cannot standard-scale a vector with zero variance")
    return (array - array.mean()) / std
