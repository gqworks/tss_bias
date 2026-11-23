"""ATCC MSA-1003 mock community: exact-strain references should show no TSS bias.

The 20 reference URLs and the Illumina/HiFi accessions are the ones in
``mock_community_analysis/download_mock.pbs``. Genomes are concatenated in
that order. Coverage bias uses the same Prodigal filter and ±500 bp metric
as the gut cohorts.
"""

from __future__ import annotations

import argparse
import shutil

from tssbias.config import Config
from tssbias.errors import MissingInputError
from tssbias.external import require_tool, run_command, run_pipeline
from tssbias.logutil import get_logger

LOG = get_logger("tssbias.mock")

# Illumina and PacBio accessions from the lab download script.
ILLUMINA = "SRR8359173"
HIFI = "SRR11606871"
REFERENCES = [
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/012/825/GCF_000012825.1_ASM1282v1/GCF_000012825.1_ASM1282v1_genomic.fna.gz",
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/010/425/GCF_000010425.1_ASM1042v1/GCF_000010425.1_ASM1042v1_genomic.fna.gz",
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/020/546/685/GCF_020546685.1_ASM2054668v1/GCF_020546685.1_ASM2054668v1_genomic.fna.gz",
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/172/575/GCF_000172575.2_ASM17257v2/GCF_000172575.2_ASM17257v2_genomic.fna.gz",
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/154/225/GCF_000154225.1_ASM15422v1/GCF_000154225.1_ASM15422v1_genomic.fna.gz",
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/013/372/085/GCF_013372085.1_ASM1337208v1/GCF_013372085.1_ASM1337208v1_genomic.fna.gz",
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/231/215/GCF_000231215.1_ASM23121v1/GCF_000231215.1_ASM23121v1_genomic.fna.gz",
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/307/795/GCF_000307795.1_ASM30779v1/GCF_000307795.1_ASM30779v1_genomic.fna.gz",
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/014/425/GCF_000014425.1_ASM1442v1/GCF_000014425.1_ASM1442v1_genomic.fna.gz",
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/008/805/GCF_000008805.1_ASM880v1/GCF_000008805.1_ASM880v1_genomic.fna.gz",
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/008/005/GCF_000008005.1_ASM800v1/GCF_000008005.1_ASM800v1_genomic.fna.gz",
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/767/745/GCF_000767745.1_ASM76774v1/GCF_000767745.1_ASM76774v1_genomic.fna.gz",
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/006/765/GCF_000006765.1_ASM676v1/GCF_000006765.1_ASM676v1_genomic.fna.gz",
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/013/465/GCF_000013465.1_ASM1346v1/GCF_000013465.1_ASM1346v1_genomic.fna.gz",
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/007/265/GCF_000007265.1_ASM726v1/GCF_000007265.1_ASM726v1_genomic.fna.gz",
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/335/055/GCF_000335055.2_ASM33505v2/GCF_000335055.2_ASM33505v2_genomic.fna.gz",
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/010/505/GCF_000010505.1_ASM1050v1/GCF_000010505.1_ASM1050v1_genomic.fna.gz",
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/015/985/GCF_000015985.1_ASM1598v1/GCF_000015985.1_ASM1598v1_genomic.fna.gz",
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/007/645/GCF_000007645.1_ASM764v1/GCF_000007645.1_ASM764v1_genomic.fna.gz",
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/007/465/GCF_000007465.2_ASM746v2/GCF_000007465.2_ASM746v2_genomic.fna.gz",
]


def run(cfg: Config, args: argparse.Namespace) -> None:
    """Align the mock Illumina sample to the merged ATCC references.

    Inputs:
        args.download: fetch the 20 genomes and the two SRA FASTQs when missing.
    Outputs:
        ``results/mock/SRR8359173_mean_bias.csv`` via the shared matrix writer.
        The matrix itself is ``results/matrices/mock_merged_SRR8359173.csv``.
    """
    del args
    mock_dir = cfg.data_root / "mock_ref"
    fasta = mock_dir / "merged_genomes.fna"
    fastq_dir = cfg.data_root / "fastq" / "mock"
    r1 = fastq_dir / f"{ILLUMINA}_1.fastq.gz"
    r2 = fastq_dir / f"{ILLUMINA}_2.fastq.gz"
    if not fasta.exists() or not r1.exists() or not r2.exists():
        raise MissingInputError(
            "mock community FASTA and Illumina FASTQs are not staged. "
            "Download the 20 ATCC genomes listed in mock_community_analysis/download_mock.pbs "
            f"into {fasta} (concatenated) and {ILLUMINA} reads into {fastq_dir}. "
            f"HiFi reads are {HIFI}. No mock bias was fabricated.",
            [fasta, r1, r2],
        )
    from tssbias.commands.discover import _one_pair
    from tssbias.config import Species
    from tssbias.coverage import read_matrix
    from tssbias.metrics import coverage_bias
    import pandas as pd

    species = Species("mock_merged", "mock_merged", "", None, ())
    # Stage the merged fasta where the discovery helpers expect an assembly.
    staged = cfg.fasta_path("mock_merged")
    staged.parent.mkdir(parents=True, exist_ok=True)
    if not staged.exists():
        shutil.copyfile(fasta, staged)
    tss_path = cfg.tss_table_path("mock_merged")
    if not tss_path.exists():
        gff = staged.parent / "mock_merged_prodigal.gff"
        cds = staged.parent / "mock_merged_prodigal.fna"
        require_tool("prodigal")
        run_command(["prodigal", "-i", str(staged), "-o", str(gff), "-d", str(cds), "-f", "gff", "-p", "meta"])
        from tssbias.tss import filter_isolated_tss, parse_prodigal_gff

        filter_isolated_tss(parse_prodigal_gff(gff), cfg.window_bp).to_csv(tss_path, index=False)
        require_tool("bowtie2-build")
        run_command(["bowtie2-build", str(staged), str(staged)])
    _one_pair(cfg, "illumina", fastq_dir, ILLUMINA, species, force=False)
    _tags, windows = read_matrix(cfg.matrix_path("mock_merged", ILLUMINA), cfg.window_bp)
    bias = coverage_bias(windows, cfg.window_bp)
    finite = bias[pd.notna(bias)]
    out = cfg.results_dir / "mock" / f"{ILLUMINA}_mean_bias.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"sample": ILLUMINA, "n_tss": int(finite.size), "mean_coverage_bias": float(finite.mean()) if finite.size else float("nan")}]
    ).to_csv(out, index=False)
    LOG.info("wrote %s", out)


def download_urls() -> list[str]:
    """Reference URLs recorded in the original mock download script."""
    return list(REFERENCES)
