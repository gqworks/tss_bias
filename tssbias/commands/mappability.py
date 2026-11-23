"""ART uniform-read simulation, the mappability control.

Reads are simulated with the methods parameters
``art_illumina -na -ss HS25 -p -l 150 -c 500000 -m200 -s 10``, aligned with
Bowtie2 default settings, and tested with the same position permutation as
real coverage. A null result here is the manuscript's mappability control;
this command does not assume that outcome.
"""

from __future__ import annotations

import argparse

import pandas as pd

from tssbias.commands.common import passing_pairs, write_table
from tssbias.commands.discover import _one_pair
from tssbias.config import Config
from tssbias.coverage import depth_from_table
from tssbias.errors import MissingInputError
from tssbias.external import require_tool, run_command
from tssbias.logutil import get_logger
from tssbias.permute import permute_coverage
from tssbias.stats import benjamini_hochberg
from tssbias.tss import read_tss_table

LOG = get_logger("tssbias.mappability")


def run(cfg: Config, args: argparse.Namespace) -> None:
    """Simulate, align, and permute one cohort.

    Outputs:
        ``results/mappability/{cohort}_simu_fdr.csv``
    """
    cohort = cfg.cohort(args.cohort)
    if cohort.seq_type != "illumina":
        raise MissingInputError("ART mappability control is defined for Illumina cohorts", [cohort.name])
    pairs = passing_pairs(cfg, cohort, args)
    require_tool("art_illumina")
    rows = []
    seen = set()
    for sample, species in pairs:
        if species.assembly in seen:
            continue
        seen.add(species.assembly)
        fasta = cfg.fasta_path(species.assembly)
        if not fasta.exists():
            raise MissingInputError("reference FASTA for ART", [fasta])
        prefix = cfg.scratch_root / "art" / species.assembly
        prefix.parent.mkdir(parents=True, exist_ok=True)
        r1 = Path_fastq(prefix, 1)
        r2 = Path_fastq(prefix, 2)
        if args.force or not r1.exists():
            run_command(["art_illumina", *cfg.art_args, "-i", str(fasta), "-o", str(prefix)])
        # ART writes {prefix}1.fq and {prefix}2.fq for paired reads.
        art_r1 = prefix.parent / f"{prefix.name}1.fq"
        art_r2 = prefix.parent / f"{prefix.name}2.fq"
        if not art_r1.exists():
            raise MissingInputError("ART did not write paired reads", [art_r1, art_r2])
        sim_sample = f"simu_{species.assembly}"
        # Reuse discovery alignment by staging the simulated reads under the names it expects.
        stage = cfg.scratch_root / "art_fastq" / species.assembly
        stage.mkdir(parents=True, exist_ok=True)
        staged_r1 = stage / f"{sim_sample}_1.fastq.gz"
        staged_r2 = stage / f"{sim_sample}_2.fastq.gz"
        if args.force or not staged_r1.exists():
            _gzip_to(art_r1, staged_r1)
            _gzip_to(art_r2, staged_r2)
        _one_pair(cfg, "illumina", stage, sim_sample, species, force=args.force)
        depth = cfg.depth_path(species.assembly, sim_sample)
        if not depth.exists():
            raise MissingInputError("simulated alignment did not write a depth table", [depth])
        result = permute_coverage(
            depth_from_table(depth),
            read_tss_table(cfg.tss_table_path(species.assembly)),
            n_permutations=cfg.n_permutations,
            seed=cfg.seed,
            sample=sim_sample,
            assembly=species.assembly,
            window_bp=cfg.window_bp,
        )
        rows.append(
            {
                "species": species.name,
                "assembly": species.assembly,
                "mean_bias": result.mean_statistic,
                "pvalue": result.pvalue,
                "n_tss": result.n_tss,
                "n_permutations": result.n_permutations,
            }
        )
    frame = pd.DataFrame(rows)
    reject, adjusted = benjamini_hochberg(frame["pvalue"].to_numpy(), cfg.fdr_alpha)
    frame["fdr_bh"] = adjusted
    frame["significant"] = reject
    destination = cfg.results_dir / "mappability" / f"{cohort.name}_simu_fdr.csv"
    write_table(destination, frame)
    LOG.info("wrote %s", destination)


def Path_fastq(prefix, end: int):
    return prefix.parent / f"{prefix.name}{end}.fq"


def _gzip_to(src, dest) -> None:
    import gzip
    import shutil

    with src.open("rb") as raw, gzip.open(dest, "wb") as zipped:
        shutil.copyfileobj(raw, zipped)
