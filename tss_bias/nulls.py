"""Permutation / simulation null helpers for TSS bias tests."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
import pyBigWig
from Bio import SeqIO

from tss_bias.bias import calc_bias, randomize, scale_mat


def get_gc_null(tss_dat: pd.DataFrame, fasta_sequences: dict) -> float:
    null_gc = []
    for chrom in list(set(tss_dat.Genome)):
        random_tss = randomize(np.array(tss_dat[tss_dat.Genome == chrom]["Start"]))
        strand_perc = sum((tss_dat.Genome == chrom) & (tss_dat.Strand == "+")) / sum(tss_dat.Genome == chrom)
        genome_len = len(fasta_sequences[chrom].seq)
        for i in random_tss:
            if i <= 501 or i >= genome_len - 500:
                continue
            left_perc = 100 * len(re.findall("G|C", str(fasta_sequences[chrom].seq[i - 500 : i]))) / 500
            right_perc = 100 * len(re.findall("G|C", str(fasta_sequences[chrom].seq[i : i + 500]))) / 500
            if np.random.binomial(1, strand_perc) == 1:
                null_gc.append(right_perc - left_perc)
            else:
                null_gc.append(left_perc - right_perc)
    return float(np.mean(null_gc))


def get_simu_null(simu_bw, tss_dat: pd.DataFrame, fasta_sequences: dict) -> pd.DataFrame:
    null_simu_mat = []
    for chrom in list(set(tss_dat.Genome)):
        random_tss = randomize(np.array(tss_dat[tss_dat.Genome == chrom]["Start"]))
        strand_perc = sum((tss_dat.Genome == chrom) & (tss_dat.Strand == "+")) / sum(tss_dat.Genome == chrom)
        genome_len = len(fasta_sequences[chrom].seq)
        for i in random_tss:
            if i <= 501 or i >= genome_len - 500:
                continue
            window = simu_bw.values(chrom, i - 501, i + 500)
            if np.random.binomial(1, strand_perc) == 0:
                window.reverse()
            null_simu_mat.append(window)
    return pd.DataFrame(null_simu_mat)


def get_fixed_null(depth, annot: pd.DataFrame) -> pd.DataFrame:
    genome_dict = depth.chroms()
    null_mat = []
    for chrom in list(set(annot.Genome)):
        random_tss = randomize(np.array(annot[annot.Genome == chrom]["Start"]))
        strand_perc = sum((annot.Genome == chrom) & (annot.Strand == "+")) / sum(annot.Genome == chrom)
        for i in random_tss:
            if i <= 500 or i >= genome_dict[chrom] - 500:
                continue
            window = depth.values(chrom, i - 501, i + 500)
            if np.random.binomial(1, strand_perc) == 0:
                window.reverse()
            null_mat.append(window)
    return pd.DataFrame(null_mat)


def get_simu_bias_mat(asm: str, refseq_dir: str, beds_dir: str, scale: bool = True) -> pd.DataFrame:
    fna = f"{refseq_dir}/{asm}/{asm}.fna"
    fasta_sequences = SeqIO.to_dict(SeqIO.parse(fna, "fasta"))
    tss_dat = pd.read_csv(f"{refseq_dir}/{asm}/{asm}_filtered_tss.csv")
    tss_dat_np = tss_dat.to_numpy()
    simu_bw = pyBigWig.open(f"{beds_dir}/{asm}_simu.bw")
    simu_mat = []
    for i in tss_dat_np:
        genome_len = len(fasta_sequences[i[0]].seq)
        start = i[5]
        if start <= 501 or start >= genome_len - 500:
            window = [0] * 1001
        else:
            window = simu_bw.values(i[0], start - 501, start + 500)
        if i[3] == "-":
            window.reverse()
        simu_mat.append(window)
    simu_mat = pd.DataFrame(simu_mat)
    simu_mat.index = [re.match(r"ID=([^;]*);.*", x[4]).group(1) for x in tss_dat_np]
    if scale:
        scaled_mat = scale_mat(simu_mat).fillna(0)
        return scaled_mat
    return simu_mat


def gene_length_from_annot(row: Any) -> int:
    return abs(int(row[2]) - int(row[1]))
