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


def test_hmp3_stool_replaces_private_hyp():
    from tssbias.cohorts import read_manifest
    from tssbias.config import load_config

    cfg = load_config()
    assert "hyp" not in cfg.cohorts
    cohort = cfg.cohort("hmp3_stool")
    assert cohort.bioproject == "PRJNA275349"
    assert cohort.abundance_min_pct == 0.5
    assert cohort.abundance_min_samples == 5
    assert cohort.species_set == "correlation"
    assert cohort.paired_rna is False
    frame = pd.read_csv(cohort.manifest, sep="\t", dtype=str)
    assert len(frame) == 89
    assert frame["sample_alias"].tolist() == sorted(frame["sample_alias"])
    assert frame["sample_alias"].str.endswith("G_DNA_Stool").all()
    assert frame["sample_id"].is_unique
    assert frame["sample_alias"].is_unique
    assert frame["biosample"].is_unique
    assert set(frame["bioproject"]) == {"PRJNA275349"}
    assert set(frame["sra_study"]) == {"SRP056641"}
    assert set(frame["library_layout"]) == {"PAIRED"}
    # Duplicate deposit of 700021123_zg_G_DNA_Stool; the smaller SRR is kept.
    assert "SRR2175754" not in set(frame["sample_id"])
    assert "SRR1952455" in set(frame["sample_id"])
    samples = read_manifest(cohort.manifest)
    assert [sample.sample_id for sample in samples] == frame["sample_id"].tolist()
    assert samples[0].sample_id == "SRR2175645"


def test_liu2016_replaces_private_arg():
    from tssbias.cohorts import read_manifest
    from tssbias.config import load_config

    cfg = load_config()
    assert "arg" not in cfg.cohorts
    cohort = cfg.cohort("liu2016")
    assert cohort.display_name == "Liu et al."
    assert cohort.bioproject == "PRJNA328899"
    assert cohort.abundance_min_pct == 0.5
    assert cohort.abundance_min_samples == 10
    assert cohort.species_set == "liu2016"
    assert cohort.paired_rna is False
    assert cohort.fastq_dir.name == "liu2016"
    frame = pd.read_csv(cohort.manifest, sep="\t", dtype=str)
    assert len(frame) == 110
    assert frame["sample_alias"].tolist() == sorted(frame["sample_alias"])
    assert frame["sample_id"].is_unique
    assert frame["sample_alias"].is_unique
    assert frame["biosample"].is_unique
    assert set(frame["bioproject"]) == {"PRJNA328899"}
    assert set(frame["sra_study"]) == {"SRP080787"}
    assert set(frame["library_layout"]) == {"PAIRED"}
    assert set(frame["library_strategy"]) == {"WGS"}
    samples = read_manifest(cohort.manifest)
    assert [sample.sample_id for sample in samples] == frame["sample_id"].tolist()
    assert samples[0].sample_id == "SRR3992955"


def test_liu2016_discover_fails_when_fastqs_are_absent(capsys):
    code = main(["discover", "--cohort", "liu2016", "--sample-index", "0"])
    captured = capsys.readouterr()
    assert code == 2
    assert "does not download FASTQs" in captured.err
    assert "fastq/liu2016/SRR3992955_1.fastq.gz" in captured.err
    assert "fastq/liu2016/SRR3992955_2.fastq.gz" in captured.err
    assert "liu2016/merged_mpa.txt" in captured.err
    assert "No results were written" in captured.err


def test_hmp3_stool_discover_fails_when_fastqs_are_absent(capsys):
    code = main(["discover", "--cohort", "hmp3_stool", "--sample-index", "0"])
    captured = capsys.readouterr()
    assert code == 2
    assert "does not download FASTQs" in captured.err
    assert "SRR2175645_1.fastq.gz" in captured.err
    assert "SRR2175645_2.fastq.gz" in captured.err
    assert "hmp3_stool/merged_mpa.txt" in captured.err
    assert "No results were written" in captured.err
