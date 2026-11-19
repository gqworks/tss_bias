"""Shared TSS coverage bias helpers (scale / bias / nulls)."""

from __future__ import annotations

import re
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def scale_mat(d: pd.DataFrame) -> pd.DataFrame:
    mean_vals = d.mean(axis=1).to_numpy()
    std_vals = d.std(axis=1).to_numpy()
    return (d - mean_vals[:, None]) / std_vals[:, None]


def calc_bias(x: Sequence[float]) -> float:
    left_sum = np.sum(x[:500])
    right_sum = np.sum(x[500:])
    return float(right_sum - left_sum)


def randomize(intervals: np.ndarray) -> np.ndarray:
    intervals = intervals.astype(int)
    point_diffs = intervals[1:] - intervals[:-1]
    max_len = intervals[-1]
    new_points = np.concatenate((np.array(point_diffs).cumsum(), [max_len]))
    new_points = (new_points + np.random.randint(0, max_len)) % max_len
    return new_points


def sort_tss_bias(x: pd.DataFrame) -> pd.Index:
    x = x.copy()
    x["bias"] = [calc_bias(i) for i in np.array(x)]
    return x.sort_values(["bias"], ascending=False).index


def abundance_norm_tpm(tpm: float, mapped_reads: float, pseudocount: float = 0.001) -> float:
    """Abundance-normalized TPM used in revision analyses."""
    norm = 1e6 * tpm / mapped_reads if mapped_reads else np.nan
    return float(np.log(norm + pseudocount))


def parse_locus_tag(desc: str) -> str:
    m = re.match(r"^ID=([^;]*)", desc)
    if not m:
        raise ValueError(f"Cannot parse locus tag from: {desc}")
    return m.group(1)


def mean_bias(mat: pd.DataFrame, scale: bool = True) -> float:
    work = scale_mat(mat).fillna(0) if scale else mat
    return float(np.mean([calc_bias(row) for row in np.array(work)]))
