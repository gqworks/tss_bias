"""Coverage windows around Prodigal TSSs.

BigWig access is imported only when a command actually opens a file, so the
metric tests do not need pyBigWig.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tssbias.metrics import coverage_bias, window_length


def orient_windows(
    depth: dict[str, np.ndarray],
    tss: pd.DataFrame,
    window_bp: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract oriented 1001-bp windows.

    Inputs:
        depth: contig name -> per-base coverage, index 0 is the first base.
        tss: frame with genome, tss (1-based), strand, locus_tag.
        window_bp: flank length.
    Outputs:
        ``(windows, covered)``. ``windows`` has shape (n, 2*window_bp+1) with
        downstream to the right. ``covered`` is False for windows that fall off
        a contig or are entirely zero; those rows are zeros in ``windows``.
    """
    n = len(tss)
    width = window_length(window_bp)
    windows = np.zeros((n, width), dtype=float)
    covered = np.zeros(n, dtype=bool)
    genomes = tss["genome"].to_numpy()
    coords = tss["tss"].to_numpy(dtype=int)
    strands = tss["strand"].to_numpy()
    for i in range(n):
        array = depth.get(str(genomes[i]))
        if array is None:
            continue
        t0 = int(coords[i]) - 1
        start = t0 - window_bp
        end = t0 + window_bp + 1
        if start < 0 or end > array.shape[0]:
            continue
        window = np.asarray(array[start:end], dtype=float)
        if strands[i] == "-":
            window = window[::-1]
        if np.any(window):
            windows[i] = window
            covered[i] = True
    return windows, covered


def bias_table(tss: pd.DataFrame, windows: np.ndarray, covered: np.ndarray, window_bp: int = 500) -> pd.DataFrame:
    """Attach coverage bias to a TSS table.

    Inputs:
        tss: filtered TSS rows aligned with ``windows``.
        windows: output of :func:`orient_windows`.
        covered: True when the window has at least one nonzero base and was in range.
        window_bp: flank length.
    Outputs:
        Copy of ``tss`` with ``coverage_bias`` and ``covered``. Uncovered rows
        have NaN bias.
    """
    bias = coverage_bias(windows, window_bp)
    bias = np.where(covered, bias, np.nan)
    out = tss.copy()
    out["coverage_bias"] = bias
    out["covered"] = covered
    return out


def depth_from_table(path: Path) -> dict[str, np.ndarray]:
    """Load ``samtools depth -aa`` output. Missing positions stay 0.

    Inputs:
        path: tab-separated contig, 1-based position, depth.
    Outputs:
        Dict of contig to per-base coverage, index 0 is the first base.
    """
    positions: dict[str, list[int]] = {}
    depths: dict[str, list[float]] = {}
    with Path(path).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            chrom, pos, depth = line.split()[:3]
            positions.setdefault(chrom, []).append(int(pos))
            depths.setdefault(chrom, []).append(float(depth))
    arrays: dict[str, np.ndarray] = {}
    for chrom, coords in positions.items():
        length = max(coords)
        array = np.zeros(length, dtype=float)
        array[np.asarray(coords, dtype=int) - 1] = np.asarray(depths[chrom], dtype=float)
        arrays[chrom] = array
    return arrays


def depth_from_bigwig(path: Path) -> dict[str, np.ndarray]:
    """Load a bigWig into per-contig float arrays. Missing bases are 0.

    Inputs:
        path: bamCoverage bigWig.
    Outputs:
        Dict of contig to coverage. Raises ImportError if pyBigWig is absent.
    """
    import pyBigWig

    handle = pyBigWig.open(str(path))
    try:
        arrays: dict[str, np.ndarray] = {}
        for chrom, length in handle.chroms().items():
            values = handle.values(chrom, 0, int(length), numpy=True)
            arrays[chrom] = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0)
        return arrays
    finally:
        handle.close()


def matrix_column_names(window_bp: int = 500) -> list[str]:
    """Stable coverage-matrix column names, TSS at ``pos_0``.

    Inputs:
        window_bp: flank length.
    Outputs:
        ``["locus_tag", "pos_-500", ..., "pos_0", ..., "pos_500"]``.
    """
    offsets = list(range(-window_bp, window_bp + 1))
    return ["locus_tag", *[f"pos_{offset}" for offset in offsets]]


def write_matrix(path: Path, locus_tags: list[str], windows: np.ndarray, window_bp: int = 500) -> None:
    """Write a TSS coverage matrix.

    Inputs:
        path: destination CSV.
        locus_tags: one id per row, same order as ``windows``.
        windows: shape (n, 2*window_bp+1).
        window_bp: flank length.
    Outputs:
        CSV with a header. Parent directories are created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(windows, columns=matrix_column_names(window_bp)[1:])
    frame.insert(0, "locus_tag", list(locus_tags))
    frame.to_csv(path, index=False)


def read_matrix(path: Path, window_bp: int = 500) -> tuple[pd.Series, np.ndarray]:
    """Read a matrix written by :func:`write_matrix`.

    Inputs:
        path: CSV path.
        window_bp: expected flank length.
    Outputs:
        ``(locus_tag, windows)``.
    """
    frame = pd.read_csv(path)
    expected = matrix_column_names(window_bp)
    if list(frame.columns) != expected:
        raise ValueError(f"{path} columns do not match a {window_length(window_bp)}-bp TSS window")
    tags = frame["locus_tag"].astype(str)
    windows = frame.drop(columns=["locus_tag"]).to_numpy(dtype=float)
    return tags, windows
