"""Short reads mapped to same-sample long reads.

Long reads shorter than 10 kb are dropped. Each sample is subsampled to
100,000 reads with the pipeline seed. Illumina reads are aligned with
Bowtie2 ``-a``. Prodigal (``-p meta``) predicts genes, ignoring calls within
500 bp of a read end. Gene sequences are matched to pinned RefSeq CDS with
minimap2 so each long-read TSS can be compared with the RefSeq bias for that
species.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from tssbias.cohorts import longread_fastq, read_manifest
from tssbias.commands.common import write_table
from tssbias.config import Config
from tssbias.coverage import depth_from_bigwig, orient_windows
from tssbias.errors import MissingInputError
from tssbias.external import require_tool, run_command, run_pipeline
from tssbias.logutil import get_logger
from tssbias.metrics import coverage_bias
from tssbias.stats import rng_for
from tssbias.tss import filter_isolated_tss, parse_prodigal_gff

LOG = get_logger("tssbias.matched-longread")


def run(cfg: Config, args: argparse.Namespace) -> None:
    """Compare RefSeq bias with same-sample long-read bias.

    Outputs:
        ``results/matched_longread/bias.csv``
        ``results/matched_longread/paired_test.csv``
    """
    cohort = cfg.cohort("matched")
    if not cohort.manifest.exists():
        raise MissingInputError("matched-sample manifest", [cohort.manifest])
    samples = read_manifest(cohort.manifest)
    if args.sample:
        samples = [sample for sample in samples if sample.sample_id == args.sample]
    if args.sample_index is not None:
        if args.sample_index < 0 or args.sample_index >= len(samples):
            raise MissingInputError(
                f"sample index {args.sample_index} is outside the matched manifest ({len(samples)} rows)",
                [cohort.manifest],
            )
        samples = [samples[args.sample_index]]
    # download writes DNA under fastq/<manifest stem>/ and long reads under longread/.
    longread_dir = cohort.fastq_dir / "longread"
    rows = []
    for sample in samples:
        if not sample.longread_id:
            raise MissingInputError(f"{sample.sample_id} has no long-read id", [cohort.manifest])
        rows.extend(_one(cfg, cohort.fastq_dir, longread_dir, sample.sample_id, sample.longread_id))
    frame = pd.DataFrame(rows)
    out = cfg.results_dir / "matched_longread"
    write_table(out / "bias.csv", frame)
    paired = frame.dropna(subset=["refseq_mean_bias", "longread_mean_bias"])
    if len(paired) >= 2:
        result = ttest_rel(paired["refseq_mean_bias"], paired["longread_mean_bias"])
        pvalue, stat = float(result.pvalue), float(result.statistic)
    else:
        pvalue, stat = float("nan"), float("nan")
    write_table(
        out / "paired_test.csv",
        pd.DataFrame([{"contrast": "refseq_vs_longread", "n": int(len(paired)), "statistic": stat, "pvalue": pvalue}]),
    )
    LOG.info("wrote matched long-read comparison for %s samples", len(frame))


def _one(cfg: Config, illumina_dir, longread_dir, sample: str, longread_id: str) -> list[dict]:
    long_fastq = longread_fastq(longread_dir, longread_id)
    r1 = illumina_dir / f"{sample}_1.fastq.gz"
    r2 = illumina_dir / f"{sample}_2.fastq.gz"
    missing = [path for path in (r1, r2) if not path.exists()]
    if long_fastq is None:
        missing.extend(
            [
                longread_dir / f"{longread_id}.fastq.gz",
                longread_dir / f"{longread_id}_1.fastq.gz",
            ]
        )
    if missing:
        raise MissingInputError(
            "matched short and long FASTQs. Stage them with "
            "`python -m tssbias download data/manifests/matched_longread.tsv`. "
            "This command does not download FASTQs.",
            missing,
        )
    work = cfg.scratch_root / "matched" / sample
    work.mkdir(parents=True, exist_ok=True)
    subset = work / "longreads.fa"
    if not subset.exists():
        _subsample(cfg, long_fastq, subset, sample)
    gff = work / "prodigal.gff"
    cds = work / "prodigal.fna"
    if not gff.exists():
        require_tool("prodigal")
        run_command(["prodigal", "-i", str(subset), "-o", str(gff), "-d", str(cds), "-f", "gff", "-p", "meta"])
    tss = filter_isolated_tss(parse_prodigal_gff(gff), cfg.window_bp)
    bam = work / "illumina.sorted.bam"
    if not bam.exists():
        require_tool("bowtie2-build")
        require_tool("bowtie2")
        run_command(["bowtie2-build", str(subset), str(subset)])
        run_pipeline(
            [
                ["bowtie2", "-a", "-p", str(cfg.threads), "-x", str(subset), "-1", str(r1), "-2", str(r2)],
                ["samtools", "view", "-b", "-"],
                ["samtools", "sort", "-@", str(cfg.threads), "-o", str(bam), "-"],
            ]
        )
        run_command(["samtools", "index", str(bam)])
    bigwig = work / "illumina.bw"
    if not bigwig.exists():
        run_command(["bamCoverage", "--bam", str(bam), "-p", str(cfg.threads), "-o", str(bigwig), "-of", "bigwig"])
    windows, covered = orient_windows(depth_from_bigwig(bigwig), tss, cfg.window_bp)
    per_gene = tss.copy()
    per_gene["coverage_bias"] = coverage_bias(windows, cfg.window_bp)
    per_gene.loc[~covered, "coverage_bias"] = np.nan
    require_tool("minimap2")
    rows = []
    for species in cfg.species_in("matched"):
        ref_cds = cfg.assembly_dir(species.assembly) / f"{species.assembly}_prodigal.fna"
        matrix = cfg.matrix_path(species.assembly, sample)
        if not ref_cds.exists() or not matrix.exists():
            raise MissingInputError(
                "matched comparison needs RefSeq CDS and the discovery matrix. "
                "Run prepare-references and discover --cohort matched.",
                [ref_cds, matrix],
            )
        loci = _loci_mapping_to(cds, ref_cds, work / f"{species.assembly}.paf")
        chosen = per_gene.loc[per_gene["locus_tag"].isin(loci), "coverage_bias"].dropna()
        from tssbias.coverage import read_matrix

        _tags, ref_windows = read_matrix(matrix, cfg.window_bp)
        ref_bias = coverage_bias(ref_windows, cfg.window_bp)
        ref_finite = ref_bias[np.isfinite(ref_bias)]
        rows.append(
            {
                "sample": sample,
                "longread_id": longread_id,
                "species": species.name,
                "n_longread_tss": int(chosen.size),
                "longread_mean_bias": float(chosen.mean()) if chosen.size else float("nan"),
                "refseq_mean_bias": float(ref_finite.mean()) if ref_finite.size else float("nan"),
            }
        )
    return rows


def _loci_mapping_to(long_cds: Path, ref_cds: Path, paf: Path) -> set[str]:
    """Locus tags of long-read genes with a minimap2 hit to a RefSeq CDS."""
    run_command(["minimap2", "-x", "asm5", str(ref_cds), str(long_cds), "-o", str(paf)])
    loci = set()
    if not paf.exists():
        return loci
    for line in paf.read_text().splitlines():
        parts = line.split("\t")
        if parts:
            loci.add(parts[0].split()[0])
    return loci


def _subsample(cfg: Config, fastq: Path, destination, sample: str) -> None:
    """Keep reads at least 10 kb, then take ``longread_subsample`` of them."""
    import gzip

    sequences = []
    opener = gzip.open if str(fastq).endswith(".gz") else open
    with opener(fastq, "rt") as handle:
        while True:
            header = handle.readline()
            if not header:
                break
            sequence = handle.readline().strip()
            plus = handle.readline()
            handle.readline()
            if not header.startswith("@") or not plus.startswith("+"):
                raise MissingInputError(f"{fastq} is not FASTQ; refusing to guess records", [fastq])
            if len(sequence) >= cfg.longread_min_read_bp:
                sequences.append((header[1:].split()[0], sequence))
    if not sequences:
        raise MissingInputError(f"no long reads >= {cfg.longread_min_read_bp} bp in {fastq}", [fastq])
    rng = rng_for(cfg.seed, "matched-subsample", sample)
    take = min(cfg.longread_subsample, len(sequences))
    chosen = rng.choice(len(sequences), size=take, replace=False)
    with destination.open("w") as handle:
        for index in np.sort(chosen):
            header, seq = sequences[int(index)]
            handle.write(f">{header}\n{seq}\n")
    LOG.info("subsampled %s long reads for %s", take, sample)
