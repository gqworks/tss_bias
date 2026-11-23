"""sylph containment ANI versus TSS coverage bias (reviewer figure).

Genome sketches and sketches of the 500 bp upstream of each filtered TSS are
queried against each sample. Adjusted ANI is correlated with mean coverage
bias. The fraction of the genome covered by the union of those upstream
windows is written beside the correlation; the manuscript discusses a figure
near 24.9% for that overlap.
"""

from __future__ import annotations

import argparse

import pandas as pd
from scipy.stats import pearsonr

from tssbias.commands.common import passing_pairs, write_table
from tssbias.config import Config
from tssbias.coverage import read_matrix
from tssbias.errors import MissingInputError
from tssbias.external import require_tool, run_command
from tssbias.fasta import read_fasta
from tssbias.logutil import get_logger
from tssbias.metrics import coverage_bias
from tssbias.cohorts import fastq_paths
from tssbias.tss import read_tss_table

LOG = get_logger("tssbias.sylph")


def run(cfg: Config, args: argparse.Namespace) -> None:
    """Sketch, query, and correlate ANI with bias.

    Outputs:
        ``results/sylph/{cohort}_ani.csv``
        ``results/sylph/{cohort}_correlation.csv``
        ``results/sylph/{cohort}_upstream_fraction.csv``
    """
    cohort = cfg.cohort(args.cohort)
    pairs = passing_pairs(cfg, cohort, args)
    require_tool("sylph")
    rows = []
    fractions = []
    seen_asm: set[str] = set()
    for sample, species in pairs:
        fasta = cfg.fasta_path(species.assembly)
        tss_path = cfg.tss_table_path(species.assembly)
        matrix = cfg.matrix_path(species.assembly, sample.sample_id)
        reads = fastq_paths(cohort.fastq_dir, sample.sample_id, cohort.seq_type)
        missing = [path for path in [fasta, tss_path, matrix, *reads] if not path.exists()]
        if missing:
            raise MissingInputError("sylph needs the reference, TSS table, coverage matrix, and FASTQs", missing)
        if species.assembly not in seen_asm:
            fractions.append(_upstream_fraction(cfg, species.assembly, species.name))
            seen_asm.add(species.assembly)
        ani = _query(cfg, fasta, reads, species.assembly, sample.sample_id)
        _tags, windows = read_matrix(matrix, cfg.window_bp)
        bias = coverage_bias(windows, cfg.window_bp)
        finite = bias[pd.notna(bias)]
        rows.append(
            {
                "sample": sample.sample_id,
                "species": species.name,
                "assembly": species.assembly,
                "adjusted_ani": ani,
                "mean_coverage_bias": float(finite.mean()) if finite.size else float("nan"),
            }
        )
    frame = pd.DataFrame(rows).dropna()
    out = cfg.results_dir / "sylph"
    write_table(out / f"{cohort.name}_ani.csv", frame)
    write_table(out / f"{cohort.name}_upstream_fraction.csv", pd.DataFrame(fractions))
    corr_rows = []
    for species_name, subset in frame.groupby("species"):
        if len(subset) < 3 or subset["adjusted_ani"].std() == 0 or subset["mean_coverage_bias"].std() == 0:
            coefficient, pvalue = float("nan"), float("nan")
        else:
            coefficient, pvalue = pearsonr(subset["adjusted_ani"], subset["mean_coverage_bias"])
        corr_rows.append(
            {
                "species": species_name,
                "n": int(len(subset)),
                "pearson_r": float(coefficient),
                "pvalue": float(pvalue),
            }
        )
    write_table(out / f"{cohort.name}_correlation.csv", pd.DataFrame(corr_rows))
    LOG.info("wrote sylph correlations for %s", cohort.name)


def _upstream_fraction(cfg: Config, assembly: str, species: str) -> dict:
    sequences = read_fasta(cfg.fasta_path(assembly))
    tss = read_tss_table(cfg.tss_table_path(assembly))
    covered = {name: set() for name in sequences}
    for row in tss.itertuples(index=False):
        seq = sequences.get(str(row.genome))
        if seq is None:
            continue
        t0 = int(row.tss) - 1
        if row.strand == "+":
            start, end = t0 - cfg.window_bp, t0
        else:
            start, end = t0 + 1, t0 + 1 + cfg.window_bp
        start = max(0, start)
        end = min(len(seq), end)
        covered[str(row.genome)].update(range(start, end))
    genome_len = sum(len(seq) for seq in sequences.values())
    covered_n = sum(len(bases) for bases in covered.values())
    return {
        "species": species,
        "assembly": assembly,
        "genome_bp": genome_len,
        "upstream_window_bp": covered_n,
        "fraction": covered_n / genome_len if genome_len else float("nan"),
    }


def _query(cfg: Config, fasta, reads, assembly: str, sample: str) -> float:
    sketch_dir = cfg.sketches_dir
    sketch_dir.mkdir(parents=True, exist_ok=True)
    genome_sketch = sketch_dir / f"{assembly}.sylsp"
    sample_sketch = sketch_dir / f"{sample}.sylsp"
    if not genome_sketch.exists():
        run_command(["sylph", "sketch", "-g", str(fasta), "-o", str(sketch_dir / assembly)])
    if not sample_sketch.exists():
        run_command(["sylph", "sketch", *[str(read) for read in reads], "-o", str(sketch_dir / sample)])
    result = sketch_dir / f"{assembly}_{sample}.tsv"
    run_command(["sylph", "query", str(genome_sketch), str(sample_sketch), "-o", str(result)])
    table = pd.read_csv(result, sep="\t")
    if "Adjusted_ANI" not in table.columns:
        raise MissingInputError(
            "sylph query output has no Adjusted_ANI column; refusing to guess a column",
            [result],
        )
    return float(table["Adjusted_ANI"].iloc[0])
