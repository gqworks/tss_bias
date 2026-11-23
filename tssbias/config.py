"""Typed run configuration.

Load order:
1. YAML file (``--config``, else ``TSS_CONFIG``, else ``config/default.yaml``).
2. Environment overrides: ``TSS_DATA_ROOT``, ``TSS_SCRATCH_ROOT``,
   ``TSS_PROJECT``, ``TSS_RESULTS_DIR``, ``TSS_SEED``.
3. ``{repo_root}``, ``{project}``, ``{data_root}``, ``{scratch_root}``,
   ``{results_dir}`` placeholders are expanded.

Species pins and cohort thresholds live in the YAML so every command shares
one definition of the 0.5% screen, the ±500 bp window, the permutation count,
and the expression bins.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from tssbias.errors import MissingInputError


@dataclass(frozen=True)
class Species:
    """One pinned RefSeq assembly."""

    name: str
    assembly: str
    ftp: str
    taxid: int | None
    sets: tuple[str, ...]


@dataclass(frozen=True)
class Cohort:
    """One manuscript cohort."""

    name: str
    manifest: Path
    fastq_dir: Path
    seq_type: str
    abundance_min_pct: float
    abundance_min_samples: int
    species_set: str
    abundance_table: Path | None
    bioproject: str
    paired_rna: bool


@dataclass(frozen=True)
class Config:
    """Resolved configuration. Paths are absolute."""

    config_path: Path
    repo_root: Path
    project: str
    queue: str
    walltime: str
    ncpus: int
    mem: str
    threads: int
    seed: int
    data_root: Path
    scratch_root: Path
    results_dir: Path
    refseq_dir: Path
    bams_dir: Path
    bigwig_dir: Path
    matrices_dir: Path
    mpa_merged: Path
    kallisto_dir: Path
    assembled_dir: Path
    promotech_dir: Path
    sketches_dir: Path
    pacbio_abundance: Path
    nanopore_abundance: Path
    window_bp: int
    n_permutations: int
    longread_permutations: int
    fdr_alpha: float
    tpm_low: float
    tpm_mid: float
    tpm_pseudocount: float
    bowtie2_mp: tuple[int, ...]
    read_length: int
    five_prime_window: int
    five_prime_upstream_shift: int
    five_prime_offsets: tuple[int, ...]
    min_gene_bp_five_prime: int
    flank_buffer_bp: int
    longread_min_read_bp: int
    longread_subsample: int
    longread_min_gene_bp: int
    mapq_min: int
    mag_min_fraction: float
    mag_hq_completeness: float
    mag_hq_contamination: float
    promotech_bp: int
    promotech_top_fraction_matches_tss_count: bool
    promotech_score_threshold: float
    art_args: tuple[str, ...]
    correlation_min_samples: int
    correlation_min_pct: float
    correlation_balance_n: int
    kraken_db: str
    metaphlan_db: str
    promotech_home: str
    figshare_url: str
    species: dict[str, Species] = field(default_factory=dict)
    cohorts: dict[str, Cohort] = field(default_factory=dict)

    def species_in(self, set_name: str) -> list[Species]:
        """Pinned species whose ``sets`` include ``set_name``, in YAML order."""
        return [item for item in self.species.values() if set_name in item.sets]

    def cohort(self, name: str) -> Cohort:
        """Return one cohort or raise MissingInputError."""
        if name not in self.cohorts:
            known = ", ".join(sorted(self.cohorts))
            raise MissingInputError(
                f"unknown cohort {name!r}. Known cohorts: {known}",
                [],
            )
        return self.cohorts[name]

    def assembly_dir(self, assembly: str) -> Path:
        return self.refseq_dir / assembly

    def fasta_path(self, assembly: str) -> Path:
        return self.assembly_dir(assembly) / f"{assembly}.fna"

    def tss_table_path(self, assembly: str) -> Path:
        return self.assembly_dir(assembly) / f"{assembly}_filtered_tss.csv"

    def matrix_path(self, assembly: str, sample: str, tag: str = "") -> Path:
        suffix = f"_{tag}" if tag else ""
        return self.matrices_dir / f"{assembly}_{sample}{suffix}.csv"

    def bam_path(self, assembly: str, sample: str, tag: str = "") -> Path:
        suffix = f"_{tag}" if tag else ""
        return self.bams_dir / f"{assembly}_{sample}{suffix}.sorted.bam"

    def bigwig_path(self, assembly: str, sample: str, tag: str = "") -> Path:
        suffix = f"_{tag}" if tag else ""
        return self.bigwig_dir / f"{assembly}_{sample}{suffix}.bw"


def find_repo_root(start: Path) -> Path:
    """Walk parents until ``config/default.yaml`` is found."""
    for candidate in [start, *start.parents]:
        if (candidate / "config" / "default.yaml").is_file():
            return candidate
    raise MissingInputError(
        "could not find config/default.yaml by walking parent directories",
        [start],
    )


def load_config(path: Path | None = None) -> Config:
    """Load and resolve configuration.

    Inputs:
        path: YAML file. None selects ``TSS_CONFIG`` or the repo default.
    Outputs:
        Config with absolute paths and expanded placeholders.
    """
    if path is None:
        env = os.environ.get("TSS_CONFIG")
        if env:
            path = Path(env)
        else:
            path = find_repo_root(Path.cwd()) / "config" / "default.yaml"
    path = Path(path).resolve()
    if not path.is_file():
        raise MissingInputError("configuration file", [path])
    raw = yaml.safe_load(path.read_text())
    repo_root = find_repo_root(path.parent)
    _apply_env(raw)
    tokens = {
        "repo_root": str(repo_root),
        "project": str(raw.get("gadi", {}).get("project", "")),
        "data_root": "",
        "scratch_root": "",
        "results_dir": "",
    }
    # Expand roots first, then the paths that refer to them.
    paths = raw.setdefault("paths", {})
    for key in ("data_root", "scratch_root", "results_dir"):
        tokens[key] = _expand(str(paths.get(key, "")), tokens)
        paths[key] = tokens[key]
    expanded = _expand_tree(raw, tokens)
    species = _load_species(repo_root / "config" / "species.yaml")
    return _build(path, repo_root, expanded, species)


def _apply_env(raw: dict[str, Any]) -> None:
    paths = raw.setdefault("paths", {})
    gadi = raw.setdefault("gadi", {})
    analysis = raw.setdefault("analysis", {})
    if os.environ.get("TSS_DATA_ROOT"):
        paths["data_root"] = os.environ["TSS_DATA_ROOT"]
    if os.environ.get("TSS_SCRATCH_ROOT"):
        paths["scratch_root"] = os.environ["TSS_SCRATCH_ROOT"]
    if os.environ.get("TSS_RESULTS_DIR"):
        paths["results_dir"] = os.environ["TSS_RESULTS_DIR"]
    if os.environ.get("TSS_PROJECT"):
        gadi["project"] = os.environ["TSS_PROJECT"]
    if os.environ.get("TSS_SEED"):
        analysis["seed"] = int(os.environ["TSS_SEED"])


def _expand(value: str, tokens: dict[str, str]) -> str:
    current = value
    for _ in range(6):
        updated = current.format(**tokens)
        if updated == current:
            return updated
        current = updated
    return current


def _expand_tree(node: Any, tokens: dict[str, str]) -> Any:
    if isinstance(node, str):
        return _expand(node, tokens)
    if isinstance(node, list):
        return [_expand_tree(item, tokens) for item in node]
    if isinstance(node, dict):
        return {key: _expand_tree(value, tokens) for key, value in node.items()}
    return node


def _load_species(path: Path) -> dict[str, Species]:
    if not path.is_file():
        raise MissingInputError("species accession table", [path])
    raw = yaml.safe_load(path.read_text())["species"]
    species: dict[str, Species] = {}
    for name, item in raw.items():
        species[name] = Species(
            name=name,
            assembly=item["assembly"],
            ftp=item["ftp"],
            taxid=None if item.get("taxid") in (None, "") else int(item["taxid"]),
            sets=tuple(item.get("sets", [])),
        )
    return species


def _build(path: Path, repo_root: Path, raw: dict[str, Any], species: dict[str, Species]) -> Config:
    gadi = raw.get("gadi", {})
    paths = raw["paths"]
    analysis = raw["analysis"]
    software = raw.get("software", {})
    cohorts: dict[str, Cohort] = {}
    for name, item in raw["cohorts"].items():
        table = item.get("abundance_table") or ""
        cohorts[name] = Cohort(
            name=name,
            manifest=Path(item["manifest"]),
            fastq_dir=Path(item["fastq_dir"]),
            seq_type=item["seq_type"],
            abundance_min_pct=float(item["abundance_min_pct"]),
            abundance_min_samples=int(item["abundance_min_samples"]),
            species_set=item["species_set"],
            abundance_table=Path(table) if table else None,
            bioproject=str(item.get("bioproject", "")),
            paired_rna=bool(item.get("paired_rna", False)),
        )
    return Config(
        config_path=path,
        repo_root=repo_root,
        project=str(gadi.get("project", "")),
        queue=str(gadi.get("queue", "normal")),
        walltime=str(gadi.get("walltime", "24:00:00")),
        ncpus=int(gadi.get("ncpus", 8)),
        mem=str(gadi.get("mem", "32GB")),
        threads=int(raw.get("compute", {}).get("threads", 8)),
        seed=int(analysis["seed"]),
        data_root=Path(paths["data_root"]),
        scratch_root=Path(paths["scratch_root"]),
        results_dir=Path(paths["results_dir"]),
        refseq_dir=Path(paths["refseq_dir"]),
        bams_dir=Path(paths["bams_dir"]),
        bigwig_dir=Path(paths["bigwig_dir"]),
        matrices_dir=Path(paths["matrices_dir"]),
        mpa_merged=Path(paths["mpa_merged"]),
        kallisto_dir=Path(paths["kallisto_dir"]),
        assembled_dir=Path(paths["assembled_dir"]),
        promotech_dir=Path(paths["promotech_dir"]),
        sketches_dir=Path(paths["sketches_dir"]),
        pacbio_abundance=Path(paths["pacbio_abundance"]),
        nanopore_abundance=Path(paths["nanopore_abundance"]),
        window_bp=int(analysis["window_bp"]),
        n_permutations=int(analysis["n_permutations"]),
        longread_permutations=int(analysis["longread_permutations"]),
        fdr_alpha=float(analysis["fdr_alpha"]),
        tpm_low=float(analysis["tpm_low"]),
        tpm_mid=float(analysis["tpm_mid"]),
        tpm_pseudocount=float(analysis["tpm_pseudocount"]),
        bowtie2_mp=tuple(int(v) for v in analysis["bowtie2_mp"]),
        read_length=int(analysis["read_length"]),
        five_prime_window=int(analysis["five_prime_window"]),
        five_prime_upstream_shift=int(analysis["five_prime_upstream_shift"]),
        five_prime_offsets=tuple(int(v) for v in analysis["five_prime_offsets"]),
        min_gene_bp_five_prime=int(analysis["min_gene_bp_five_prime"]),
        flank_buffer_bp=int(analysis["flank_buffer_bp"]),
        longread_min_read_bp=int(analysis["longread_min_read_bp"]),
        longread_subsample=int(analysis["longread_subsample"]),
        longread_min_gene_bp=int(analysis["longread_min_gene_bp"]),
        mapq_min=int(analysis["mapq_min"]),
        mag_min_fraction=float(analysis["mag_min_fraction"]),
        mag_hq_completeness=float(analysis["mag_hq_completeness"]),
        mag_hq_contamination=float(analysis["mag_hq_contamination"]),
        promotech_bp=int(analysis["promotech_bp"]),
        promotech_top_fraction_matches_tss_count=True,
        promotech_score_threshold=float(analysis["promotech_score_threshold"]),
        art_args=tuple(analysis["art_args"]),
        correlation_min_samples=int(analysis["correlation_min_samples"]),
        correlation_min_pct=float(analysis["correlation_min_pct"]),
        correlation_balance_n=int(analysis["correlation_balance_n"]),
        kraken_db=str(software.get("kraken_db", "")),
        metaphlan_db=str(software.get("metaphlan_db", "")),
        promotech_home=str(software.get("promotech_home", "")),
        figshare_url=str(raw.get("figshare", {}).get("project_url", "")),
        species=species,
        cohorts=cohorts,
    )
