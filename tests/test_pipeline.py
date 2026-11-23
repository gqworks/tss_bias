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
    assert "mpa4_res/hmp2/merged_mpa.txt" in captured.err
    assert "No results were written" in captured.err


def test_hmp3_stool_replaces_private_hyp():
    from tssbias.cohorts import read_manifest
    from tssbias.config import load_config

    cfg = load_config()
    assert "hyp" not in cfg.cohorts
    cohort = cfg.cohort("hmp3_stool")
    assert cohort.bioproject == "PRJNA275349"
    assert cohort.abundance_min_pct == 0.5
    assert cohort.abundance_min_samples == 30
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
    assert cohort.abundance_min_samples == 30
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


PUBLIC_COLUMNS = [
    "sample_id",
    "sample_alias",
    "biosample",
    "bioproject",
    "sra_study",
    "library_layout",
    "library_strategy",
]


def test_public_manifests_use_sra_run_ids():
    from tssbias.cohorts import read_manifest
    from tssbias.config import load_config

    cfg = load_config()
    expected = {
        "hmp2": (78, "PRJNA389280", "SRP115812", True),
        "franzosa": (8, "PRJNA188481", "SRP019038", False),
        "pacbio": (11, "PRJNA754443", "SRP332749", False),
        "nanopore": (20, "PRJNA820119", "SRP366030", False),
    }
    for name, (n, project, study, paired) in expected.items():
        cohort = cfg.cohort(name)
        assert cohort.bioproject == project
        assert cohort.paired_rna is paired
        frame = pd.read_csv(cohort.manifest, sep="\t", dtype=str)
        assert list(frame.columns[:7]) == PUBLIC_COLUMNS
        assert len(frame) == n
        assert frame["sample_alias"].tolist() == sorted(frame["sample_alias"])
        assert frame["sample_id"].is_unique
        assert frame["sample_alias"].is_unique
        assert frame["biosample"].is_unique
        assert set(frame["bioproject"]) == {project}
        assert set(frame["sra_study"]) == {study}
        text = cohort.manifest.read_text()
        assert "/groups/" not in text
        assert "cgsd" not in text
        samples = read_manifest(cohort.manifest)
        assert [sample.sample_id for sample in samples] == frame["sample_id"].tolist()
    franzosa = pd.read_csv(cfg.cohort("franzosa").manifest, sep="\t", dtype=str)
    assert franzosa["sample_alias"].str.endswith("_RNAlater").all()
    assert set(franzosa["library_strategy"]) == {"WGS"}
    hmp2 = pd.read_csv(cfg.cohort("hmp2").manifest, sep="\t", dtype=str)
    assert hmp2["rna_id"].is_unique
    assert set(hmp2["rna_id"]).isdisjoint(set(hmp2["sample_id"]))
    nanopore = pd.read_csv(cfg.cohort("nanopore").manifest, sep="\t", dtype=str)
    assert set(nanopore["sample_alias"]) == {f"CD{i}" for i in range(81, 101)}
    assert set(nanopore["library_layout"]) == {"SINGLE"}
    assert nanopore["shortread_srr"].is_unique
    assert set(nanopore["shortread_srr"]).isdisjoint(set(nanopore["sample_id"]))
    ont = read_manifest(cfg.cohort("nanopore").manifest)
    assert [sample.shortread_id for sample in ont] == nanopore["shortread_srr"].tolist()
    matched = cfg.cohort("matched")
    assert matched.manifest == cfg.cohort("nanopore").manifest
    assert matched.sample_column == "shortread_srr"
    assert matched.longread_column == "sample_id"
    pairs = read_manifest(matched.manifest, matched.sample_column, matched.longread_column)
    assert [sample.sample_id for sample in pairs] == nanopore["shortread_srr"].tolist()
    assert [sample.longread_id for sample in pairs] == nanopore["sample_id"].tolist()
    assert not (cfg.repo_root / "data/manifests/matched_longread.tsv").exists()


def test_hmp3_stool_discover_fails_when_fastqs_are_absent(capsys):
    code = main(["discover", "--cohort", "hmp3_stool", "--sample-index", "0"])
    captured = capsys.readouterr()
    assert code == 2
    assert "does not download FASTQs" in captured.err
    assert "SRR2175645_1.fastq.gz" in captured.err
    assert "SRR2175645_2.fastq.gz" in captured.err
    assert "hmp3_stool/merged_mpa.txt" in captured.err
    assert "No results were written" in captured.err


def test_mock_cohort_uses_one_manifest_and_does_not_download(capsys):
    from tssbias.config import load_config

    cfg = load_config()
    cohort = cfg.cohort("mock")
    frame = pd.read_csv(cohort.manifest, sep="\t", dtype=str)
    assert list(frame["sample_id"]) == ["SRR8359173"]
    assert list(frame["longread_srr"]) == ["SRR11606871"]
    assert cohort.fastq_dir.name == "mock"
    assert cfg.cohort("matched").fastq_dir.name == "shortread"
    code = main(["mock"])
    captured = capsys.readouterr()
    assert code == 2
    assert "download data/manifests/mock.tsv" in captured.err
    assert "does not download FASTQs" in captured.err
    code = main(["mock", "--hifi"])
    captured = capsys.readouterr()
    assert code == 2
    assert "fastq/mock/longread/SRR11606871" in captured.err
    code = main(["five-prime", "--cohort", "mock"])
    captured = capsys.readouterr()
    assert code == 2
    assert "mock_merged" in captured.err
    code = main(["matched-longread", "--sample-index", "0"])
    captured = capsys.readouterr()
    assert code == 2
    assert "fastq/nanopore/shortread/SRR18491235" in captured.err
    assert "fastq/nanopore/SRR18490957" in captured.err


def test_hifi_subset_is_100k_and_written_before_downstream(tmp_path):
    from tssbias.commands.mock import hifi_subset_path, write_hifi_subset
    from tssbias.config import load_config
    from tssbias.stats import rng_for

    cfg = load_config()
    assert cfg.longread_subsample == 100_000
    subset = hifi_subset_path(cfg)
    assert subset == cfg.scratch_root / "mock" / "hifi" / "SRR11606871_100k.fa"
    fastq = tmp_path / "hifi.fastq"
    records = []
    with fastq.open("w") as handle:
        for index in range(12):
            name = f"read{index}"
            records.append(name)
            handle.write(f"@{name} extra\n{'A' * (20 + index)}\n+\n{'I' * (20 + index)}\n")
    destination = tmp_path / "SRR11606871_100k.fa"
    written = write_hifi_subset(fastq, destination, 4, rng_for(1, "mock-hifi-100k", "SRR11606871"))
    assert written == 4
    names = [line[1:].split()[0] for line in destination.read_text().splitlines() if line.startswith(">")]
    assert len(names) == 4
    assert set(names) <= set(records)
    again = tmp_path / "again.fa"
    write_hifi_subset(fastq, again, 4, rng_for(1, "mock-hifi-100k", "SRR11606871"))
    assert again.read_text() == destination.read_text()
    assert not destination.with_name(destination.name + ".partial").exists()


def test_abundance_tables_are_metaphlan4_outputs():
    from tssbias.config import load_config

    cfg = load_config()
    assert not (cfg.repo_root / "metaphlan3_res").exists()
    assert not (cfg.repo_root / "mock_community_analysis" / "metaphlan3_res").exists()
    for name, cohort in cfg.cohorts.items():
        assert cohort.abundance_table == cfg.data_root / "mpa4_res" / name / "merged_mpa.txt"
    assert cfg.mpa_merged == cfg.cohort("hmp2").abundance_table


def test_metaphlan_fails_when_fastqs_are_absent(capsys):
    code = main(["metaphlan", "data/manifests/hmp2.tsv"])
    captured = capsys.readouterr()
    assert code == 2
    assert "does not download FASTQs" in captured.err
    assert "fastq/hmp2/SRR5947813_1.fastq.gz" in captured.err
    assert "fastq/hmp2/SRR5947813_2.fastq.gz" in captured.err
    assert "No results were written" in captured.err
    code = main(["metaphlan", "data/manifests/pacbio.tsv", "--sample-index", "0"])
    captured = capsys.readouterr()
    assert code == 2
    assert "fastq/pacbio/SRR15489020.fastq.gz" in captured.err
    code = main(["metaphlan", "data/manifests/does-not-exist.tsv"])
    captured = capsys.readouterr()
    assert code == 2
    assert "does-not-exist.tsv" in captured.err


def test_metaphlan_merge_keeps_species_columns(tmp_path):
    from tssbias.commands.metaphlan import _merge

    one = tmp_path / "s1.txt"
    two = tmp_path / "s2.txt"
    one.write_text(
        "clade_name\tNCBI_tax_id\trelative_abundance\n"
        "k__Bacteria|s__A\t1\t10\n"
        "k__Bacteria|s__B\t2\t1\n"
    )
    two.write_text("clade_name\trelative_abundance\nk__Bacteria|s__A\t4\nk__Bacteria|s__C\t6\n")
    merged = _merge([one, two], ["s1", "s2"])
    assert list(merged.columns) == ["clade_name", "s1", "s2"]
    row = merged.set_index("clade_name")
    assert row.loc["k__Bacteria|s__A", "s1"] == 10
    assert row.loc["k__Bacteria|s__A", "s2"] == 4
    assert row.loc["k__Bacteria|s__C", "s1"] == 0


def test_species_screen_uses_the_approved_thresholds():
    from tssbias.cohorts import species_passing
    from tssbias.config import load_config

    cfg = load_config()
    assert not (cfg.repo_root / "config" / "species.yaml").exists()
    expected = {
        "hmp2": 30,
        "franzosa": 5,
        "hmp3_stool": 30,
        "liu2016": 30,
        "nanopore": 4,
        "pacbio": 4,
        "matched": 5,
    }
    for name, minimum in expected.items():
        cohort = cfg.cohort(name)
        assert cohort.abundance_min_pct == 0.5
        assert cohort.abundance_min_samples == minimum
    samples = [f"s{i}" for i in range(30)]
    rows = {
        "s__Kept": [0.5] * 30,
        "s__Borderline": [0.5] * 29 + [0.49],
        "s__Low": [0.49] * 30,
    }
    abundance = pd.DataFrame(rows, index=samples).T
    assert species_passing(abundance, samples, 0.5, 30) == ["s__Kept"]
    assert species_passing(abundance, samples, 0.5, 5) == ["s__Borderline", "s__Kept"]


def test_refseq_choice_prefers_reference_genome(tmp_path):
    from tssbias.config import Species
    from tssbias.refseq import assembly_from_summary, binomial, choose_assembly, read_resolved, write_resolved

    assert binomial("s__Phocaeicola_vulgatus") == "Phocaeicola vulgatus"
    representative = {
        "assemblyaccession": "GCF_000000001.1",
        "assemblyname": "rep",
        "ftppath_refseq": "ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/000/001/GCF_000000001.1_rep",
        "refseq_category": "representative genome",
        "taxid": "1",
    }
    reference = {
        "assemblyaccession": "GCF_000000009.1",
        "assemblyname": "ref",
        "ftppath_refseq": "ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/000/009/GCF_000000009.1_ref",
        "refseq_category": "reference genome",
        "taxid": "9",
    }
    chosen = choose_assembly([representative, reference])
    assert chosen["assemblyaccession"] == "GCF_000000009.1"
    assembly, ftp, taxid = assembly_from_summary(chosen)
    assert assembly == "GCF_000000009.1_ref"
    assert ftp.startswith("https://ftp.ncbi.nlm.nih.gov/")
    assert taxid == 9
    path = tmp_path / "resolved_accessions.tsv"
    write_resolved(path, {"s__A": Species("s__A", assembly, ftp, taxid), "s__B": Species("s__B", "", "", None)})
    loaded = read_resolved(path)
    assert loaded["s__A"].assembly == assembly
    assert loaded["s__B"].assembly == ""
    assert loaded["s__B"].taxid is None
