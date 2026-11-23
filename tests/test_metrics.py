"""Metric definitions from the manuscript."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tssbias.metrics import (
    abundance_normalised_tpm,
    coverage_bias,
    expression_group,
    gc_bias_from_sequence,
    window_length,
)
from tssbias.stats import benjamini_hochberg, greater_permutation_pvalue, shift_tss_positions
from tssbias.tss import filter_isolated_tss


def test_window_and_positive_bias():
    window = np.zeros((1, window_length(500)))
    window[0, 501:] = 5
    window[0, :500] = 1
    window[0, 500] = 3
    bias = coverage_bias(window, 500)
    assert bias.shape == (1,)
    assert bias[0] > 0


def test_negative_bias_when_upstream_is_higher():
    window = np.zeros((1, 1001))
    window[0, :500] = 4
    window[0, 501:] = 1
    assert coverage_bias(window, 500)[0] < 0


def test_zero_coverage_is_undefined():
    assert np.isnan(coverage_bias(np.zeros((2, 1001)), 500)).all()


def test_expression_bins_and_normalisation():
    tpm = np.array([0.0, 0.1, 0.05, 1.0, 2.0])
    # mapped_reads = 1000 → norm = 1e6 * tpm / 1000 = 1000 * tpm
    # → 0, 100, 50, 1000, 2000
    norm, log_norm = abundance_normalised_tpm(tpm, mapped_reads=1000, pseudocount=0.001)
    assert norm.tolist() == pytest.approx([0, 100, 50, 1000, 2000])
    assert log_norm[0] == pytest.approx(np.log(0.001))
    groups = expression_group(norm, low=100, mid=1000)
    assert list(groups) == ["zero", "mid", "low", "high", "high"]
    # 0 < x < 100 is low. 50 is low. 100 is mid.
    assert list(expression_group(np.array([0, 50, 100, 999, 1000]))) == [
        "zero",
        "low",
        "mid",
        "mid",
        "high",
    ]


def test_gc_bias_strand():
    upstream = "A" * 500
    downstream = "G" * 500
    sequence = upstream + "T" + downstream
    # TSS is the middle T, 1-based index 501.
    assert gc_bias_from_sequence(sequence, 501, "+", 500) == pytest.approx(100)
    assert gc_bias_from_sequence(sequence, 501, "-", 500) == pytest.approx(-100)


def test_tss_filter_requires_more_than_500_bp():
    genes = pd.DataFrame(
        {
            "genome": ["c", "c", "c"],
            "gene_left": [100, 600, 2000],
            "gene_right": [200, 700, 2200],
            "strand": ["+", "+", "+"],
            "desc": ["ID=a", "ID=b", "ID=c"],
            "tss": [100, 600, 2000],
            "locus_tag": ["a", "b", "c"],
            "gene_length": [101, 101, 201],
        }
    )
    kept = filter_isolated_tss(genes, 500)
    # 100 and 600 are 500 bp apart, so both drop. 2000 is >500 from 600.
    assert kept["locus_tag"].tolist() == ["c"]


def test_permutation_pvalue_matches_lab_estimator():
    assert greater_permutation_pvalue(5, np.array([0, 1, 2, 3])) == pytest.approx(0.25)
    assert greater_permutation_pvalue(0, np.array([1, 2, 3, 4])) == pytest.approx(1.0)


def test_shift_is_deterministic_and_preserves_count():
    rng_a = np.random.default_rng(1)
    rng_b = np.random.default_rng(1)
    positions = np.array([1000, 2500, 8000])
    assert shift_tss_positions(positions, rng_a).tolist() == shift_tss_positions(positions, rng_b).tolist()
    assert len(shift_tss_positions(positions, np.random.default_rng(2))) == 3


def test_bh_rejects_small_pvalues():
    reject, adjusted = benjamini_hochberg(np.array([0.001, 0.8, np.nan]), alpha=0.05)
    assert reject[0]
    assert not reject[1]
    assert not reject[2]
    assert np.isnan(adjusted[2])
