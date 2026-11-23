"""MAG remapping (Suppl. Fig. 5).

Each sample is assembled with MEGAHIT, binned with MetaBAT2, and classified
with Kraken2. The bin whose species-level Kraken fraction is highest is kept
when that fraction is at least 80%. TSS bias on that bin is compared with the
RefSeq bias. A CheckM2 table, when present, flags high-quality bins
(completeness > 90, contamination < 5) but those bins are not required: the
manuscript reports that the HQ filter did not remove the residual bias.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import ttest_rel

from tssbias.commands.common import passing_pairs, write_table
from tssbias.config import Config, Species
from tssbias.coverage import depth_from_bigwig, orient_windows
from tssbias.errors import MissingInputError
from tssbias.external import require_tool, run_command, run_pipeline
from tssbias.logutil import get_logger
from tssbias.metrics import coverage_bias
from tssbias.cohorts import fastq_paths

LOG = get_logger("tssbias.mag")


def run(cfg: Config, args: argparse.Namespace) -> None:
    """Assemble, choose bins, and compare MAG bias with RefSeq bias.

    Outputs:
        ``results/mag/{cohort}_bias.csv``
        ``results/mag/{cohort}_paired_test.csv``
    """
    cohort = cfg.cohort(args.cohort)
    if not cfg.kraken_db:
        raise MissingInputError(
            "Kraken2 database is unset. Set software.kraken_db in the config.",
            [cfg.config_path],
        )
    pairs = passing_pairs(cfg, cohort, args)
    require_tool("megahit")
    require_tool("metabat2")
    require_tool("kraken2")
    rows = []
    for sample, species in pairs:
        if species.taxid is None:
            raise MissingInputError(
                f"{species.name} has no taxid in the RefSeq accession cache. "
                "Re-run prepare-references --force for this cohort.",
                [cfg.refseq_dir / "resolved_accessions.tsv"],
            )
        bin_fasta = _best_bin(cfg, cohort.fastq_dir, sample.sample_id, species)
        if bin_fasta is None:
            LOG.info("no bin >= %s%% for %s in %s", cfg.mag_min_fraction, species.name, sample.sample_id)
            continue
        mag_bias = _bias_on_fasta(cfg, cohort.fastq_dir, sample.sample_id, species, bin_fasta, args.force)
        ref_bias = _refseq_mean(cfg, sample.sample_id, species)
        rows.append(
            {
                "sample": sample.sample_id,
                "species": species.name,
                "taxid": species.taxid,
                "bin_fasta": str(bin_fasta),
                "mag_mean_bias": mag_bias,
                "refseq_mean_bias": ref_bias,
                "high_quality": _is_hq(cfg, bin_fasta),
            }
        )
    frame = pd.DataFrame(rows)
    out = cfg.results_dir / "mag" / cohort.name
    if not frame.empty:
        for sample_id, subset in frame.groupby("sample"):
            write_table(out / f"{sample_id}.csv", subset)
    if args.sample is not None or args.sample_index is not None:
        if frame.empty:
            LOG.info("no Kraken bin reached %.0f%% for this sample", cfg.mag_min_fraction)
        return
    shards = [pd.read_csv(path) for path in sorted(out.glob("*.csv"))]
    frame = pd.concat(shards, ignore_index=True) if shards else frame
    if frame.empty:
        raise MissingInputError(
            "no Kraken bin reached the 80% species fraction. Assembly outputs are under assembled_dir.",
            [cfg.assembled_dir],
        )
    out = cfg.results_dir / "mag"
    write_table(out / f"{cohort.name}_bias.csv", frame)
    paired = frame.dropna(subset=["mag_mean_bias", "refseq_mean_bias"])
    if len(paired) >= 2:
        result = ttest_rel(paired["refseq_mean_bias"], paired["mag_mean_bias"])
        pvalue, stat, n = float(result.pvalue), float(result.statistic), int(len(paired))
    else:
        pvalue, stat, n = float("nan"), float("nan"), int(len(paired))
    write_table(
        out / f"{cohort.name}_paired_test.csv",
        pd.DataFrame([{"contrast": "refseq_vs_mag", "n": n, "statistic": stat, "pvalue": pvalue}]),
    )
    LOG.info("wrote MAG comparison for %s pairs", len(frame))


def _best_bin(cfg: Config, fastq_dir, sample: str, species: Species) -> Path | None:
    sample_dir = cfg.assembled_dir / sample
    contigs = sample_dir / "final.contigs.fa"
    if not contigs.exists():
        r1, r2 = fastq_paths(fastq_dir, sample, "illumina")
        run_command(
            ["megahit", "-t", str(cfg.threads), "-1", str(r1), "-2", str(r2), "-o", str(sample_dir)]
        )
    depth = sample_dir / "depth.txt"
    if not depth.exists():
        run_command(["bowtie2-build", str(contigs), str(contigs)])
        bam = sample_dir / "contigs.bam"
        r1, r2 = fastq_paths(fastq_dir, sample, "illumina")
        run_pipeline(
            [
                ["bowtie2", "-p", str(cfg.threads), "-x", str(contigs), "-1", str(r1), "-2", str(r2)],
                ["samtools", "view", "-b", "-"],
                ["samtools", "sort", "-@", str(cfg.threads), "-o", str(bam), "-"],
            ]
        )
        run_command(["samtools", "index", str(bam)])
        require_tool("jgi_summarize_bam_contig_depths")
        run_command(["jgi_summarize_bam_contig_depths", "--outputDepth", str(depth), str(bam)])
    bins = sample_dir / "bins"
    if not any(bins.glob("*.fa")):
        bins.mkdir(parents=True, exist_ok=True)
        run_command(["metabat2", "-i", str(contigs), "-a", str(depth), "-o", str(bins / "bin")])
    reports = sample_dir / "kraken"
    reports.mkdir(parents=True, exist_ok=True)
    best_fraction = -1.0
    best_fasta: Path | None = None
    for fasta in sorted(bins.glob("*.fa")):
        report = reports / f"{fasta.stem}.report"
        if not report.exists():
            run_command(
                [
                    "kraken2",
                    "--db",
                    cfg.kraken_db,
                    "--threads",
                    str(cfg.threads),
                    "--report",
                    str(report),
                    "--output",
                    str(reports / f"{fasta.stem}.out"),
                    str(fasta),
                ]
            )
        fraction, taxid = _top_species(report)
        if taxid == species.taxid and fraction > best_fraction:
            best_fraction = fraction
            best_fasta = fasta
    if best_fasta is None or best_fraction < cfg.mag_min_fraction:
        return None
    return best_fasta


def _top_species(report: Path) -> tuple[float, int | None]:
    best = (0.0, None)
    for line in report.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) < 5 or parts[3] != "S":
            continue
        fraction = float(parts[0])
        if fraction > best[0]:
            best = (fraction, int(parts[4]))
    return best


def _bias_on_fasta(cfg: Config, fastq_dir, sample: str, species: Species, fasta: Path, force: bool) -> float:
    """Treat the bin as an assembly: Prodigal, align, mean bias."""
    work = fasta.parent / f"{fasta.stem}_tss"
    work.mkdir(parents=True, exist_ok=True)
    gff = work / "prodigal.gff"
    if force or not gff.exists():
        require_tool("prodigal")
        run_command(["prodigal", "-i", str(fasta), "-o", str(gff), "-f", "gff", "-p", "meta"])
    from tssbias.tss import filter_isolated_tss, parse_prodigal_gff

    tss = filter_isolated_tss(parse_prodigal_gff(gff), cfg.window_bp)
    if tss.empty:
        return float("nan")
    bam = work / "reads.sorted.bam"
    if force or not bam.exists():
        require_tool("bowtie2-build")
        run_command(["bowtie2-build", str(fasta), str(fasta)])
        r1, r2 = fastq_paths(fastq_dir, sample, "illumina")
        run_pipeline(
            [
                ["bowtie2", "-p", str(cfg.threads), "-x", str(fasta), "-1", str(r1), "-2", str(r2)],
                ["samtools", "view", "-b", "-"],
                ["samtools", "sort", "-@", str(cfg.threads), "-o", str(bam), "-"],
            ]
        )
        run_command(["samtools", "index", str(bam)])
    bigwig = work / "reads.bw"
    if force or not bigwig.exists():
        run_command(["bamCoverage", "--bam", str(bam), "-p", str(cfg.threads), "-o", str(bigwig), "-of", "bigwig"])
    windows, covered = orient_windows(depth_from_bigwig(bigwig), tss, cfg.window_bp)
    bias = coverage_bias(windows, cfg.window_bp)
    finite = bias[covered]
    finite = finite[pd.notna(finite)]
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())


def _refseq_mean(cfg: Config, sample: str, species: Species) -> float:
    matrix = cfg.matrix_path(species.assembly, sample)
    if not matrix.exists():
        raise MissingInputError("RefSeq coverage matrix missing. Run discover first.", [matrix])
    from tssbias.coverage import read_matrix

    _tags, windows = read_matrix(matrix, cfg.window_bp)
    bias = coverage_bias(windows, cfg.window_bp)
    finite = bias[pd.notna(bias)]
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())


def _is_hq(cfg: Config, fasta: Path) -> bool:
    """True when a sibling CheckM2 table clears the HQ thresholds.

    The table is optional. Missing CheckM2 output returns False and does not
    invent a quality call. Expected columns: completeness, contamination, bin.
    """
    table = fasta.parent.parent / "checkm2.tsv"
    if not table.exists():
        return False
    quality = pd.read_csv(table, sep="\t")
    quality.columns = [column.lower() for column in quality.columns]
    if "bin" not in quality.columns:
        return False
    hit = quality.loc[quality["bin"].astype(str) == fasta.stem]
    if hit.empty:
        return False
    row = hit.iloc[0]
    return float(row["completeness"]) > cfg.mag_hq_completeness and float(row["contamination"]) < cfg.mag_hq_contamination
