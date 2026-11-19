#!/usr/bin/env python3
"""Null / permutation tests, coverage heatmaps, long-read validation.

Stages (enable explicitly — discovery Illumina block is no longer commented-out-only):
  discovery   HMP2 real-read position-permutation null + FDR + example heatmaps
  gc          GC-content nulls for HMP2 pinned genomes
  simu        ART simulated uniform-read nulls + heatmaps
  longread    PacBio/ONT gene-length-filtered FDR (legacy live path)
  all         discovery + gc + simu + longread

Examples:
  python plot_heatmap_scatter.py --stage discovery
  python plot_heatmap_scatter.py --stage all --permutations 100
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tss_bias.config import ensure_workdir_dirs, load_config


def plot_heatmap(mat, title, vmax, dpi=300):
    import matplotlib.pyplot as plt
    import seaborn as sns

    g = sns.clustermap(
        mat,
        col_cluster=False,
        row_cluster=False,
        xticklabels=False,
        yticklabels=False,
        figsize=(10, 25),
        vmax=vmax,
        cmap="Oranges",
    )
    Path(title).parent.mkdir(parents=True, exist_ok=True)
    g.savefig(title, dpi=dpi)
    plt.close("all")


def _paths(cfg):
    return cfg["paths"]["refseq_dir"], cfg["paths"]["beds_dir"], cfg["paths"]["matrices_dir"], cfg["paths"]["plots_dir"]


def stage_gc(cfg, hmp2_genomes, n_perm):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from Bio import SeqIO

    from tss_bias.nulls import get_gc_null

    refseq, _, _, plots = _paths(cfg)
    out = Path(plots) / "null_distributions"
    out.mkdir(parents=True, exist_ok=True)
    all_pvals = []
    for species, asm in hmp2_genomes.items():
        fna = f"{refseq}/{asm}/{asm}.fna"
        fasta_sequences = SeqIO.to_dict(SeqIO.parse(fna, "fasta"))
        tss_dat = pd.read_csv(f"{refseq}/{asm}/{asm}_filtered_tss.csv")
        tss_dat_np = tss_dat.to_numpy()
        gc_bias = []
        for i in tss_dat_np:
            genome_len = len(fasta_sequences[i[0]].seq)
            start = i[5]
            strand = i[3]
            if start <= 501 or start >= genome_len - 500:
                continue
            left_perc = 100 * len(re.findall("G|C", str(fasta_sequences[i[0]].seq[start - 500 : start]))) / 500
            right_perc = 100 * len(re.findall("G|C", str(fasta_sequences[i[0]].seq[start : start + 500]))) / 500
            gc_bias.append(right_perc - left_perc if strand == "+" else left_perc - right_perc)
        real_bias = np.mean(gc_bias)
        null_bias = [get_gc_null(tss_dat, fasta_sequences) for _ in range(n_perm)]
        pval = (sum(np.array(null_bias) > real_bias) + 1) / len(null_bias)
        all_pvals.append([asm, pval])
        plt.suptitle("GC null distribution")
        plt.hist(null_bias, bins=50)
        plt.axvline(x=real_bias, color="r")
        plt.text(1, 10, f"p={pval}", size=12, color="black", weight="bold")
        plt.savefig(out / f"{asm}_gc.pdf")
        plt.clf()
    pd.DataFrame(all_pvals).to_csv(out / "hmp2_genome_pvals_gc.csv")


def stage_simu(cfg, hmp2_genomes, n_perm, threads):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import pyBigWig
    from Bio import SeqIO
    from tqdm import tqdm

    from tss_bias.bias import calc_bias, scale_mat, sort_tss_bias
    from tss_bias.nulls import get_simu_bias_mat, get_simu_null

    refseq, beds, _, plots = _paths(cfg)
    out = Path(plots) / "null_distributions"
    heat = Path(plots) / "coverage_heatmaps"
    out.mkdir(parents=True, exist_ok=True)
    heat.mkdir(parents=True, exist_ok=True)
    bams = Path(cfg["paths"]["bams_dir"])
    bams.mkdir(parents=True, exist_ok=True)
    art_mod = cfg.get("software", {}).get("art_module", "art")

    for species, asm in hmp2_genomes.items():
        fna_file = f"{refseq}/{asm}/{asm}.fna"
        if not os.path.exists(f"{refseq}/{asm}/pe_simu1.fq"):
            # Prefer bare art_illumina if module system differs on Gadi
            cmd = (
                f"module load {art_mod} 2>/dev/null; art_illumina -na -ss HS25 -i {fna_file} "
                f"-p -l 150 --fcov 30 -m 200 -s 10 -o {refseq}/{asm}/pe_simu"
            )
            subprocess.call(cmd, shell=True)
        if not os.path.exists(bams / f"{asm}_simu.sorted.bam"):
            subprocess.call(
                f"bowtie2 -p {threads} -x {fna_file} -1 {refseq}/{asm}/pe_simu1.fq -2 {refseq}/{asm}/pe_simu2.fq "
                f"| samtools view --threads {threads} -b - "
                f"| samtools sort - -o {bams}/{asm}_simu.sorted.bam --threads {threads}",
                shell=True,
            )
            subprocess.call(f"samtools index {bams}/{asm}_simu.sorted.bam -@ {threads}", shell=True)
        if not os.path.exists(f"{beds}/{asm}_simu.bw"):
            subprocess.call(
                f"bamCoverage --bam {bams}/{asm}_simu.sorted.bam -p {threads} -o {beds}/{asm}_simu.bw -of bigwig",
                shell=True,
            )

    all_pvals = []
    for species, asm in hmp2_genomes.items():
        scaled_mat = get_simu_bias_mat(asm, refseq, beds)
        simu_bias = list(map(calc_bias, np.array(scaled_mat)))
        simu_bw = pyBigWig.open(f"{beds}/{asm}_simu.bw")
        tss_dat = pd.read_csv(f"{refseq}/{asm}/{asm}_filtered_tss.csv")
        fasta_sequences = SeqIO.to_dict(SeqIO.parse(f"{refseq}/{asm}/{asm}.fna", "fasta"))
        null_bias = []
        for _ in tqdm(range(n_perm), desc=f"simu null {asm}"):
            null_mat = get_simu_null(simu_bw, tss_dat, fasta_sequences)
            scaled_null = scale_mat(null_mat).fillna(0)
            null_bias.append(np.mean(list(map(calc_bias, np.array(scaled_null)))))
        pval = (sum(np.array(null_bias) > np.mean(simu_bias)) + 1) / len(null_bias)
        all_pvals.append([asm, pval])
        plt.suptitle("Simulated reads null distribution")
        plt.hist(null_bias, bins=50)
        plt.axvline(x=np.mean(simu_bias), color="r")
        plt.savefig(out / f"{asm}_simu.pdf")
        plt.clf()
    pd.DataFrame(all_pvals).to_csv(out / "hmp2_pvals_simu.csv")

    # Example heatmaps (F. prausnitzii)
    asm = "GCF_000154385.1_ASM15438v1"
    mat = get_simu_bias_mat(asm, refseq, beds, scale=False)
    scaled_mat = scale_mat(mat).fillna(0)
    sorted_mat = mat.loc[sort_tss_bias(scaled_mat), :]
    vmax = np.percentile(sorted_mat.to_numpy().flatten(), 95)
    plot_heatmap(sorted_mat, str(heat / f"{asm}_simu.png"), vmax=vmax)
    simu_bw = pyBigWig.open(f"{beds}/{asm}_simu.bw")
    tss_dat = pd.read_csv(f"{refseq}/{asm}/{asm}_filtered_tss.csv")
    fasta_sequences = SeqIO.to_dict(SeqIO.parse(f"{refseq}/{asm}/{asm}.fna", "fasta"))
    simu_null_mat = get_simu_null(simu_bw, tss_dat, fasta_sequences)
    scaled_null = scale_mat(simu_null_mat).fillna(0)
    sorted_null = simu_null_mat.loc[sort_tss_bias(scaled_null), :]
    plot_heatmap(sorted_null, str(heat / f"{asm}_simu_null.png"), vmax=vmax)


def stage_discovery(cfg, hmp2_mpa, hmp2_genomes, n_perm):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import pyBigWig
    import seaborn as sns
    import statsmodels.stats.multitest as multi
    from scipy.stats import pearsonr

    from tss_bias.bias import calc_bias, scale_mat, sort_tss_bias
    from tss_bias.nulls import get_fixed_null, get_simu_bias_mat

    refseq, beds, matrices, plots = _paths(cfg)
    out = Path(plots) / "null_distributions"
    heat = Path(plots) / "coverage_heatmaps"
    out.mkdir(parents=True, exist_ok=True)
    heat.mkdir(parents=True, exist_ok=True)

    all_pvals = []
    for sample in hmp2_mpa.columns:
        for species in hmp2_mpa.index:
            if np.isnan(hmp2_mpa.loc[species, sample]):
                continue
            asm = hmp2_genomes[species]
            mat_path = f"{matrices}/{asm}_{sample}.csv"
            if not os.path.exists(mat_path):
                print(f"WARNING: missing matrix {mat_path}; skip")
                continue
            mat = pd.read_table(mat_path, sep=",", header=None, index_col=0)
            scaled_mat = scale_mat(mat).fillna(0)
            real_bias = list(map(calc_bias, np.array(scaled_mat)))
            null_bias = []
            tss_dat = pd.read_csv(f"{refseq}/{asm}/{asm}_filtered_tss.csv")
            bw_path = f"{beds}/{asm}_{sample}.bw"
            if not os.path.exists(bw_path):
                print(f"WARNING: missing bigWig {bw_path}; skip nulls")
                continue
            bw_depth = pyBigWig.open(bw_path)
            for _ in range(n_perm):
                null_mat = get_fixed_null(bw_depth, tss_dat)
                scaled_null = scale_mat(null_mat).fillna(0)
                null_bias.append(np.mean(list(map(calc_bias, np.array(scaled_null)))))
            pval = (sum(np.array(null_bias) > np.mean(real_bias)) + 1) / len(null_bias)
            plt.suptitle(f"{sample} {asm}")
            plt.hist(null_bias, bins=50)
            plt.axvline(x=np.mean(real_bias), color="r")
            plt.savefig(out / f"{asm}_{sample}.pdf")
            plt.clf()
            simu_scaled_mat = get_simu_bias_mat(asm, refseq, beds)
            simu_bias = list(map(calc_bias, np.array(simu_scaled_mat)))
            simu_corr = pearsonr(simu_bias, real_bias)
            all_pvals.append([asm, sample, pval, simu_corr])

    all_pvals = pd.DataFrame(all_pvals)
    all_pvals.to_csv(out / "hmp2_pvals_sample.csv")
    if all_pvals.empty:
        print("No discovery p-values computed (missing matrices/beds?).")
        return

    all_pvals = pd.read_csv(
        out / "hmp2_pvals_sample.csv", index_col=0, header=0, names=["asm", "sample", "pval", "pearson"]
    )
    all_pvals["corr"] = [float(re.match(r"\(([^,]*),.*", x).group(1)) for x in all_pvals["pearson"]]
    all_pvals["species"] = [
        list(hmp2_genomes.keys())[list(hmp2_genomes.values()).index(x)] for x in all_pvals["asm"]
    ]
    all_pvals = all_pvals.groupby(["sample", "species"]).mean(numeric_only=True).reset_index()
    all_pvals["adj_pval"] = multi.multipletests(all_pvals.pval, method="fdr_bh", alpha=0.05)[1]

    g = sns.boxplot(data=all_pvals, y="species", x="adj_pval", orient="h", color="tab:blue", showfliers=False)
    plt.xlabel("p-value (log-scaled)")
    plt.xscale("log")
    plt.axvline(0.05, color="r")
    g.figure.savefig(out / "hmp2_species_pval_boxplot.pdf", bbox_inches="tight")
    plt.clf()

    g = sns.boxplot(data=all_pvals, y="species", x="corr", orient="h", color="tab:blue")
    g.figure.savefig(out / "hmp2_species_simu_corr_boxplot.pdf", bbox_inches="tight")
    plt.clf()

    sample = "SRR5947807"
    asm = "GCF_000154385.1_ASM15438v1"
    mat_path = f"{matrices}/{asm}_{sample}.csv"
    if os.path.exists(mat_path):
        mat = pd.read_table(mat_path, sep=",", header=None, index_col=0)
        scaled_mat = scale_mat(mat).fillna(0)
        sorted_mat = mat.loc[sort_tss_bias(scaled_mat), :]
        sorted_mat = sorted_mat[sorted_mat.sum(axis=1) > 0]
        vmax = np.percentile(sorted_mat.to_numpy().flatten(), 95)
        plot_heatmap(sorted_mat, str(heat / f"{asm}_{sample}.png"), vmax=vmax, dpi=300)
        bw_depth = pyBigWig.open(f"{beds}/{asm}_{sample}.bw")
        tss_dat = pd.read_csv(f"{refseq}/{asm}/{asm}_filtered_tss.csv")
        null_mat = get_fixed_null(bw_depth, tss_dat)
        scaled_null = scale_mat(null_mat).fillna(0)
        sorted_null = null_mat.loc[sort_tss_bias(scaled_null), :]
        plot_heatmap(sorted_null, str(heat / f"{asm}_{sample}_null.png"), vmax=vmax, dpi=300)


def stage_longread(cfg, hmp2_mpa, n_perm):
    """Long-read validation with gene-length >1 kb filter (legacy live path)."""
    import numpy as np
    import pandas as pd
    import pyBigWig
    import statsmodels.stats.multitest as multi

    from tss_bias.bias import calc_bias, scale_mat
    from tss_bias.io import load_longread_abundance, load_sample_to_fastq_dir
    from tss_bias.nulls import get_fixed_null
    from tss_bias.species import genomes_for_cohort

    refseq, beds, matrices, plots = _paths(cfg)
    out = Path(plots) / "null_distributions"
    out.mkdir(parents=True, exist_ok=True)

    pacbio = load_sample_to_fastq_dir(cfg, "pacbio")
    nanopore = load_sample_to_fastq_dir(cfg, "nanopore")
    hmp2 = load_sample_to_fastq_dir(cfg, "hmp2")
    pacbio_abund = load_longread_abundance(cfg, "pacbio", list(pacbio))
    nanopore_abund = load_longread_abundance(cfg, "nanopore", list(nanopore))
    merged_abund = pd.concat([hmp2_mpa, pacbio_abund, nanopore_abund], axis=1)
    merged_abund = merged_abund.loc[
        ((merged_abund[list(hmp2)] > 0.5).sum(axis=1) >= 30)
        & ((merged_abund[list(pacbio)] > 0.5).sum(axis=1) >= 4)
        & ((merged_abund[list(nanopore)] > 0.5).sum(axis=1) >= 4),
        :,
    ]
    merged_abund = merged_abund.where(merged_abund > 0.5, np.nan)
    selected_genomes = genomes_for_cohort("longread")
    longread_samples = list(pacbio) + list(nanopore)

    all_LR_pvals = []
    for sample in longread_samples:
        for species in merged_abund.index:
            if species not in selected_genomes:
                continue
            if sample not in merged_abund.columns or np.isnan(merged_abund.loc[species, sample]):
                continue
            asm = selected_genomes[species]
            mat_path = f"{matrices}/{asm}_{sample}.csv"
            if not os.path.exists(mat_path):
                print(f"WARNING: missing {mat_path}")
                continue
            mat = pd.read_table(mat_path, sep=",", header=None, index_col=0)
            tss_dat = pd.read_csv(f"{refseq}/{asm}/{asm}_filtered_tss.csv")
            # gene length filter > 1 kb
            tss_dat["gene_len"] = (tss_dat["geneRight"] - tss_dat["geneLeft"]).abs()
            long_genes = tss_dat[tss_dat["gene_len"] > cfg["analysis"]["gene_length_min_longread"]]
            locus = [
                re.match(r"^ID=([^;]*)", d).group(1) if isinstance(d, str) else d for d in long_genes["Desc"]
            ]
            # Desc may be full attribute string
            locus = []
            for d in long_genes["Desc"]:
                m = re.match(r"^ID=([^;]*)", str(d))
                if m:
                    locus.append(m.group(1))
            keep = [i for i in mat.index if i in set(locus)]
            if not keep:
                # fall back: use all rows if parsing fails
                keep = list(mat.index)
            mat_long = mat.loc[keep]
            scaled_mat = scale_mat(mat_long).fillna(0)
            real_bias = list(map(calc_bias, np.array(scaled_mat)))
            bw_path = f"{beds}/{asm}_{sample}.bw"
            if not os.path.exists(bw_path):
                continue
            bw_depth = pyBigWig.open(bw_path)
            null_bias = []
            for _ in range(n_perm):
                null_mat = get_fixed_null(bw_depth, long_genes if len(long_genes) else tss_dat)
                scaled_null = scale_mat(null_mat).fillna(0)
                null_bias.append(np.mean(list(map(calc_bias, np.array(scaled_null)))))
            pval = (sum(np.array(null_bias) > np.mean(real_bias)) + 1) / len(null_bias)
            all_LR_pvals.append([asm, sample, pval])

    lr = pd.DataFrame(all_LR_pvals, columns=["asm", "sample", "pval"])
    lr.to_csv(out / "longread_pvals_sample.csv")
    if not lr.empty:
        lr["adj_pval"] = multi.multipletests(lr.pval, method="fdr_bh", alpha=0.05)[1]
        lr.to_csv(out / "longread_pvals_sample_fdr.csv")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument(
        "--stage",
        action="append",
        dest="stages",
        choices=["discovery", "gc", "simu", "longread", "all"],
        help="Analysis stage(s). Default if omitted: discovery (primary Illumina).",
    )
    ap.add_argument("--permutations", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    ensure_workdir_dirs(cfg)
    stages = args.stages or ["discovery"]
    if "all" in stages:
        stages = ["gc", "simu", "discovery", "longread"]

    from tss_bias.io import filter_species_abundance, load_mpa_species_table, load_sample_to_fastq_dir
    from tss_bias.species import genomes_for_cohort

    n_perm = args.permutations or int(cfg["analysis"]["permutations"])
    n_perm_lr = args.permutations or int(cfg["analysis"]["longread_permutations"])
    threads = args.threads or int(cfg["compute"]["threads"])

    hmp2_genomes = genomes_for_cohort("hmp2")
    hmp2_samples = load_sample_to_fastq_dir(cfg, "hmp2")
    mpa = load_mpa_species_table(cfg)
    c = cfg["cohorts"]["hmp2"]
    hmp2_mpa = filter_species_abundance(
        mpa, list(hmp2_samples), c["abundance_min_pct"], c["abundance_min_samples"]
    )

    for stage in stages:
        print(f"=== stage: {stage} ===")
        if stage == "gc":
            stage_gc(cfg, hmp2_genomes, n_perm)
        elif stage == "simu":
            stage_simu(cfg, hmp2_genomes, n_perm, threads)
        elif stage == "discovery":
            stage_discovery(cfg, hmp2_mpa, hmp2_genomes, n_perm)
        elif stage == "longread":
            stage_longread(cfg, hmp2_mpa, n_perm_lr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
