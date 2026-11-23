"""Upstream mate alignment scores and the alignment-score mixed model.

Mates whose mapped end falls in the 50 bp immediately downstream of the TSS
are aligned semi-globally (glocal, DNAFULL), which is the ``align`` package
call used for Fig. 5. The target sequence is the 500 bp upstream flank from
the manuscript, not the lab script's -300..+200 window. The median score per
TSS is the fixed effect in::

    scale(coverage_bias) ~ scale(upstream_alignment_score) + (1 | sample) + (1 | species)

A positive control repeats the score 200 bp into the gene. A negative control
aligns the same number of randomly chosen reads (pipeline seed) to the
upstream flank.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from tssbias.commands.common import passing_pairs, write_table
from tssbias.config import Config
from tssbias.coverage import read_matrix
from tssbias.errors import MissingInputError, ToolError
from tssbias.fasta import read_fasta
from tssbias.lmm import fit_crossed_lmm
from tssbias.logutil import get_logger
from tssbias.metrics import ALIGNMENT_LMM_FORMULA, coverage_bias
from tssbias.stats import rng_for
from tssbias.tss import read_tss_table

LOG = get_logger("tssbias.upstream")


def run(cfg: Config, args: argparse.Namespace) -> None:
    """Score upstream mates and fit the alignment mixed model.

    Outputs:
        ``results/upstream/{cohort}_scores.csv``
        ``results/upstream/{cohort}_alignment_lmm.txt``
        ``results/upstream/{cohort}_alignment_coefficients.csv``
    """
    try:
        from align.calign import aligner
        from align.matrix import DNAFULL
    except ImportError as exc:
        raise ToolError(
            "The Python package providing align.calign and align.matrix.DNAFULL is not installed. "
            "That is the semi-global aligner used for the upstream-mate scores. "
            "No alignment scores were invented."
        ) from exc
    import pysam

    cohort = cfg.cohort(args.cohort)
    pairs = passing_pairs(cfg, cohort, args)
    rows = []
    for sample, species in pairs:
        bam_path = cfg.bam_path(species.assembly, sample.sample_id)
        matrix = cfg.matrix_path(species.assembly, sample.sample_id)
        fasta = cfg.fasta_path(species.assembly)
        tss_path = cfg.tss_table_path(species.assembly)
        missing = [path for path in (bam_path, matrix, fasta, tss_path) if not path.exists()]
        if missing:
            raise MissingInputError("upstream scores need the BAM, matrix, FASTA, and TSS table", missing)
        sequences = read_fasta(fasta)
        tss = read_tss_table(tss_path)
        tags, windows = read_matrix(matrix, cfg.window_bp)
        bias = pd.Series(coverage_bias(windows, cfg.window_bp), index=tags.to_numpy())
        rng = rng_for(cfg.seed, "upstream", sample.sample_id, species.assembly)
        with pysam.AlignmentFile(str(bam_path), "rb") as bam:
            pool = _read_pool(bam, rng)
            for row in tss.itertuples(index=False):
                seq = sequences.get(str(row.genome))
                if seq is None or row.locus_tag not in bias.index:
                    continue
                target = _upstream_seq(seq, int(row.tss), str(row.strand), cfg.window_bp)
                if target is None:
                    continue
                score = _median_score(bam, row, target, aligner, DNAFULL, cfg.read_length)
                positive = _median_score(
                    bam, _shifted(row, 200), target, aligner, DNAFULL, cfg.read_length
                )
                negative = _negative_score(pool, target, score, aligner, DNAFULL, rng)
                rows.append(
                    {
                        "sample": sample.sample_id,
                        "species": species.name,
                        "locus_tag": row.locus_tag,
                        "coverage_bias": float(bias.at[row.locus_tag]),
                        "upstream_alignment_score": score,
                        "positive_control_score": positive,
                        "negative_control_score": negative,
                    }
                )
    frame = pd.DataFrame(rows).dropna(subset=["coverage_bias", "upstream_alignment_score"])
    if frame.empty:
        raise MissingInputError("no upstream alignment scores were produced", [cfg.bams_dir])
    out = cfg.results_dir / "upstream"
    write_table(out / f"{cohort.name}_scores.csv", frame)
    if frame["sample"].nunique() < 2 or frame["species"].nunique() < 2:
        LOG.info("alignment scores written; mixed model needs at least 2 samples and 2 species")
        return
    result = fit_crossed_lmm(
        frame,
        formula="bias_z ~ score_z",
        scale_columns={"coverage_bias": "bias_z", "upstream_alignment_score": "score_z"},
    )
    text = out / f"{cohort.name}_alignment_lmm.txt"
    text.write_text(
        "Manuscript formula:\n"
        f"  {ALIGNMENT_LMM_FORMULA}\n"
        "Scores are median semi-global alignment scores of 5' mates against the "
        "500 bp upstream flank. REML, Wald z.\n\n"
        + result.summary
        + "\n"
    )
    write_table(
        out / f"{cohort.name}_alignment_coefficients.csv",
        pd.DataFrame(
            {
                "term": list(result.params),
                "estimate": list(result.params.values()),
                "std_error": [result.bse[name] for name in result.params],
                "wald_pvalue": [result.pvalues[name] for name in result.params],
            }
        ),
    )
    LOG.info("wrote %s", text)


def _upstream_seq(sequence: str, tss: int, strand: str, window_bp: int) -> str | None:
    t0 = tss - 1
    if strand == "+":
        start, end = t0 - window_bp, t0
    else:
        start, end = t0 + 1, t0 + 1 + window_bp
    if start < 0 or end > len(sequence):
        return None
    return sequence[start:end]


def _shifted(row, distance: int):
    data = row._asdict()
    data["tss"] = int(row.tss) + distance if row.strand == "+" else int(row.tss) - distance
    return type(row)(**data)


def _median_score(bam, row, target: str, aligner, matrix, read_length: int) -> float:
    start, end = _mate_window(int(row.tss), str(row.strand), read_length)
    scores = []
    try:
        fetched = bam.fetch(str(row.genome), max(0, start), max(end, start + 1))
    except ValueError:
        return float("nan")
    for read in fetched:
        if read.is_unmapped or read.reference_start < start or read.reference_start >= end:
            continue
        take = (read.is_reverse and row.strand == "+") or ((not read.is_reverse) and row.strand == "-")
        if not take:
            continue
        mate_seq = _mate_sequence(bam, read, row.strand)
        if not mate_seq:
            continue
        hit = aligner(mate_seq, target, method="glocal", max_hits=1, matrix=matrix)
        if hit:
            scores.append(float(hit[0][-1]))
    if not scores:
        return float("nan")
    return float(np.median(scores))


def _mate_window(tss: int, strand: str, read_length: int) -> tuple[int, int]:
    """50 bp window of mapped 3' ends just downstream of the TSS."""
    if strand == "+":
        return tss - 5, tss + 50
    return tss - read_length - 50, tss - read_length + 5


def _mate_sequence(bam, read, strand: str) -> str | None:
    if not read.mate_is_unmapped:
        try:
            return bam.mate(read).query_sequence
        except ValueError:
            return None
    return None


def _read_pool(bam, rng) -> list[str]:
    sequences = []
    for read in bam.fetch(until_eof=True):
        if read.query_sequence and not read.is_secondary and not read.is_supplementary:
            sequences.append(read.query_sequence)
        if len(sequences) >= 5000:
            break
    rng.shuffle(sequences)
    return sequences


def _negative_score(pool: list[str], target: str, observed: float, aligner, matrix, rng) -> float:
    del observed
    if not pool:
        return float("nan")
    take = min(20, len(pool))
    choice = rng.choice(len(pool), size=take, replace=False)
    scores = []
    for index in choice:
        hit = aligner(pool[int(index)], target, method="glocal", max_hits=1, matrix=matrix)
        if hit:
            scores.append(float(hit[0][-1]))
    if not scores:
        return float("nan")
    return float(np.median(scores))
