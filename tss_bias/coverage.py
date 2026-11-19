"""Genome prep + alignment + TSS coverage matrix generation."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from tss_bias.paths import bam_path, bed_bw_path, genome_dir, matrix_path


def run_cmd(cmd: str, dry_run: bool = False) -> int:
    print(cmd)
    if dry_run:
        return 0
    return subprocess.call(cmd, shell=True)


def prepare_species_folder(
    ftp: str,
    cfg: Mapping[str, Any],
    threads: Optional[int] = None,
    dry_run: bool = False,
) -> str:
    """Download RefSeq genome, Prodigal + TSS filter, Bowtie2 index."""
    asm = ftp.rsplit("/", 1)[1]
    tax_path = genome_dir(cfg, asm)
    threads = threads or int(cfg.get("compute", {}).get("threads", 8))
    if not tax_path.exists():
        tax_path.mkdir(parents=True, exist_ok=True)
        run_cmd(f"wget {ftp}/{asm}_genomic.fna.gz -O {tax_path}/{asm}.fna.gz", dry_run=dry_run)
        run_cmd(f"gzip -df {tax_path}/{asm}.fna.gz", dry_run=dry_run)
        run_cmd(
            f"prodigal -i {tax_path}/{asm}.fna -o {tax_path}/{asm}_prodigal.gff "
            f"-d {tax_path}/{asm}_prodigal.fna -f gff",
            dry_run=dry_run,
        )
        if dry_run:
            return asm

        tss_dat = pd.read_csv(f"{tax_path}/{asm}_prodigal.gff", sep="\t", header=None, comment="#")
        tss_dat.drop(columns=[1, 2, 5, 7], inplace=True)
        tss_dat["TSS"] = [x[3] if x[6] == "+" else x[4] for i, x in tss_dat.iterrows()]
        tss_dat = np.array(tss_dat)
        non_overlap = []
        for chrom in list(set(tss_dat[:, 0])):
            subset = tss_dat[np.in1d(tss_dat[:, 0], chrom)]
            if len(subset) == 1:
                non_overlap.append(subset[0])
                continue
            if float(subset[0, 5]) + 500 < float(subset[1, 5]):
                non_overlap.append(subset[0])
            for i in range(1, len(subset) - 1):
                if float(subset[i, 5]) - 500 > float(subset[i - 1, 5]) and float(subset[i, 5]) + 500 < float(
                    subset[i + 1, 5]
                ):
                    non_overlap.append(subset[i])
            if float(subset[len(subset) - 1, 5]) - 500 > float(subset[len(subset) - 2, 5]):
                non_overlap.append(subset[len(subset) - 1])
        non_overlap = pd.DataFrame(
            non_overlap, columns=["Genome", "geneLeft", "geneRight", "Strand", "Desc", "Start"]
        )
        non_overlap.to_csv(f"{tax_path}/{asm}_filtered_tss.csv", index=False)
        genome_fna = f"{tax_path}/{asm}.fna"
        run_cmd(f"bowtie2-build {genome_fna} {genome_fna}", dry_run=dry_run)
    return asm


def calc_cov(
    asm: str,
    sample: str,
    dna_loc: str,
    seq_type: str,
    cfg: Mapping[str, Any],
    threads: Optional[int] = None,
    dry_run: bool = False,
) -> int:
    """Align reads, bamCoverage → bigWig, write ±500 bp TSS matrix."""
    threads = threads or int(cfg.get("compute", {}).get("threads", 8))
    bam = bam_path(cfg, asm, sample)
    bw = bed_bw_path(cfg, asm, sample)
    mat = matrix_path(cfg, asm, sample)
    ref_fna = genome_dir(cfg, asm) / f"{asm}.fna"
    bam.parent.mkdir(parents=True, exist_ok=True)
    bw.parent.mkdir(parents=True, exist_ok=True)
    mat.parent.mkdir(parents=True, exist_ok=True)

    if not bam.exists():
        if seq_type == "pacbio":
            if not (genome_dir(cfg, asm) / f"{asm}.fna.bwt").exists():
                run_cmd(f"bwa index {ref_fna}", dry_run=dry_run)
            run_cmd(
                f"bwa mem -t {threads} -x pacbio {ref_fna} {dna_loc}/{sample}.fastq.gz "
                f"| samtools view --threads {threads} -b - "
                f"| samtools sort - -o {bam} --threads {threads}",
                dry_run=dry_run,
            )
        elif seq_type == "nanopore":
            if not (genome_dir(cfg, asm) / f"{asm}.fna.bwt").exists():
                run_cmd(f"bwa index {ref_fna}", dry_run=dry_run)
            run_cmd(
                f"bwa mem -t {threads} -x ont2d {ref_fna} {dna_loc}/{sample}.fastq.gz "
                f"| samtools view --threads {threads} -b - "
                f"| samtools sort - -o {bam} --threads {threads}",
                dry_run=dry_run,
            )
        elif seq_type == "illumina":
            run_cmd(
                f"bowtie2 -p {threads} -x {ref_fna} "
                f"-1 {dna_loc}/{sample}_1.fastq.gz -2 {dna_loc}/{sample}_2.fastq.gz "
                f"| samtools view --threads {threads} -b - "
                f"| samtools sort - -o {bam} --threads {threads}",
                dry_run=dry_run,
            )
        else:
            raise ValueError(f"Invalid sequence type: {seq_type}")
        run_cmd(f"samtools index {bam} -@ {threads}", dry_run=dry_run)

    if not bw.exists():
        run_cmd(
            f"bamCoverage --bam {bam} -p {threads} -o {bw} -of bigwig",
            dry_run=dry_run,
        )

    if not mat.exists() and not dry_run:
        import pyBigWig

        bp_depth = pyBigWig.open(str(bw))
        genome_dict = bp_depth.chroms()
        tss_dat = pd.read_csv(genome_dir(cfg, asm) / f"{asm}_filtered_tss.csv", header=0).to_numpy()
        locus_tag = np.array([re.match(r"^ID=([^;]*)", x).group(1) for x in tss_dat[:, 4]])
        rows = []
        for row in tss_dat:
            chrom = row[0]
            if (row[5] <= 550) or (row[5] + 550 >= genome_dict[chrom]):
                window = [0] * 1001
            else:
                window = bp_depth.values(chrom, int(row[5]) - 501, int(row[5]) + 500)
                if row[3] == "-":
                    window.reverse()
            rows.append(window)
        x = np.hstack((locus_tag[np.newaxis].T, np.array(rows)))
        np.savetxt(mat, x, fmt="%s", delimiter=",")
    elif not mat.exists() and dry_run:
        print(f"[dry-run] would write {mat}")
    return 0


def run_cohort_coverage(
    cfg: Mapping[str, Any],
    cohort: str,
    sample_to_dir: Mapping[str, str],
    abundance: pd.DataFrame,
    selected_ftps: Mapping[str, list[str]],
    seq_type: str = "illumina",
    threads: Optional[int] = None,
    dry_run: bool = False,
    sample_seq_types: Optional[Mapping[str, str]] = None,
) -> None:
    """Loop sample × species for a filtered abundance table (replaces triple-quoted blocks)."""
    for sample in abundance.columns:
        dna_loc = sample_to_dir[sample]
        st = (sample_seq_types or {}).get(sample, seq_type)
        for species in abundance.index:
            if np.isnan(abundance.loc[species, sample]):
                continue
            if species not in selected_ftps:
                print(f"WARNING: no pinned genome for {species}; skipping")
                continue
            for ftp in selected_ftps[species]:
                asm = prepare_species_folder(ftp, cfg, threads=threads, dry_run=dry_run)
                calc_cov(asm, sample, dna_loc, st, cfg, threads=threads, dry_run=dry_run)
