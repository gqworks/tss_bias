"""Permutation, mixed model, and the discovery command's missing-input path."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tssbias.cli import main
from tssbias.coverage import read_matrix, write_matrix
from tssbias.lmm import fit_crossed_lmm
from tssbias.permute import permute_coverage


def test_matrix_roundtrip(tmp_path):
    windows = np.arange(6, dtype=float).reshape(2, 3)
    # window_bp 1 → length 3
    path = tmp_path / "m.csv"
    write_matrix(path, ["g1", "g2"], windows, window_bp=1)
    tags, loaded = read_matrix(path, window_bp=1)
    assert tags.tolist() == ["g1", "g2"]
    assert loaded.shape == (2, 3)


def test_permutation_detects_a_planted_downstream_peak():
    depth = {"c": np.ones(5000)}
    # Real TSSs sit on a block of elevated downstream coverage.
    for tss in (1000, 2000, 3000):
        depth["c"][tss : tss + 400] = 20
    tss = pd.DataFrame(
        {
            "genome": ["c", "c", "c"],
            "tss": [1000, 2000, 3000],
            "strand": ["+", "+", "+"],
            "locus_tag": ["a", "b", "c"],
            "gene_length": [800, 800, 800],
        }
    )
    result = permute_coverage(
        depth,
        tss,
        n_permutations=30,
        seed=1,
        sample="s",
        assembly="a",
        window_bp=100,
    )
    assert result.mean_statistic > 0
    assert result.pvalue < 0.2
    assert result.n_tss == 3
    assert result.n_permutations > 0


def test_gc_lmm_recovers_a_positive_bias_term():
    rng = np.random.default_rng(1)
    rows = []
    for sample in range(4):
        sample_effect = rng.normal()
        for species in range(3):
            species_effect = rng.normal()
            for _gene in range(25):
                bias = rng.normal()
                gc = rng.normal()
                y = 1.2 * bias + 0.1 * gc + sample_effect + species_effect + rng.normal(scale=0.2)
                rows.append(
                    {
                        "sample": f"s{sample}",
                        "species": f"p{species}",
                        "log_norm_tpm": y,
                        "coverage_bias": bias,
                        "gc_bias": gc,
                    }
                )
    result = fit_crossed_lmm(
        pd.DataFrame(rows),
        "expr_z ~ bias_z + gc_z",
        {"log_norm_tpm": "expr_z", "coverage_bias": "bias_z", "gc_bias": "gc_z"},
    )
    assert result.converged
    assert result.params["bias_z"] > 0.5
    assert result.pvalues["bias_z"] < 1e-6


def test_discover_fails_clearly_without_gadi_inputs(capsys):
    code = main(["discover", "--cohort", "hmp2", "--sample-index", "0"])
    captured = capsys.readouterr()
    assert code == 2
    assert "Missing inputs" in captured.err
    assert "merged_mpa.txt" in captured.err
    assert "No results were written" in captured.err
