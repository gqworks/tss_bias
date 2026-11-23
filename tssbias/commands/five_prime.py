"""5' versus 3' metagenomic read ends just downstream of the TSS.

The window is 50 bp starting 10 bp upstream of the TSS (manuscript), then
shifted downstream by each configured offset. Paired counting requires both
mates to map. Single counting also keeps pairs with an unmapped mate. On the
plus strand a forward read is counted as 5'; on the minus strand a reverse
read is counted as 5'. The stabilised ratio is (n5 + 1) / (n3 + 1), which is
what the lab plots used when a count is zero. Genes shorter than 500 bp are
dropped so an offset of 400 bp stays inside the gene.
"""

from __future__ import annotations

import argparse

import pandas as pd
from scipy.stats import ttest_ind, ttest_rel

from tssbias.commands.common import passing_pairs, write_table
from tssbias.config import Config
from tssbias.errors import MissingInputError
from tssbias.logutil import get_logger
from tssbias.tss import read_tss_table

LOG = get_logger("tssbias.five-prime")


def run(cfg: Config, args: argparse.Namespace) -> None:
    """Count 5' and 3' ends and write ratios plus the reported contrasts.

    Outputs:
        ``results/five_prime/{cohort}_ratios.csv``
        ``results/five_prime/{cohort}_tests.csv``
    """
    import pysam

    cohort = cfg.cohort(args.cohort)
    pairs = passing_pairs(cfg, cohort, args)
    rows = []
    for sample, species in pairs:
        bam_path = cfg.bam_path(species.assembly, sample.sample_id)
        tss_path = cfg.tss_table_path(species.assembly)
        if not bam_path.exists() or not tss_path.exists():
            raise MissingInputError("5' enrichment needs the discovery BAM and TSS table", [bam_path, tss_path])
        tss = read_tss_table(tss_path)
        tss = tss.loc[tss["gene_length"] >= cfg.min_gene_bp_five_prime]
        with pysam.AlignmentFile(str(bam_path), "rb") as bam:
            for offset in cfg.five_prime_offsets:
                for paired_only, mode in ((True, "paired"), (False, "single")):
                    n5, n3 = _count_cohort_window(bam, tss, offset, cfg, paired_only)
                    rows.append(
                        {
                            "sample": sample.sample_id,
                            "species": species.name,
                            "offset": offset,
                            "mode": mode,
                            "n5": n5,
                            "n3": n3,
                            "ratio": (n5 / n3) if n3 else float("nan"),
                            "stabilised_ratio": (n5 + 1) / (n3 + 1),
                        }
                    )
    frame = pd.DataFrame(rows)
    out = cfg.results_dir / "five_prime"
    write_table(out / f"{cohort.name}_ratios.csv", frame)
    write_table(out / f"{cohort.name}_tests.csv", _tests(frame))
    LOG.info("wrote 5' ratios for %s", cohort.name)


def _count_cohort_window(bam, tss: pd.DataFrame, offset: int, cfg: Config, paired_only: bool) -> tuple[int, int]:
    n5 = n3 = 0
    window = cfg.five_prime_window
    shift = cfg.five_prime_upstream_shift
    for row in tss.itertuples(index=False):
        start, end = _interval(int(row.tss), str(row.strand), offset, window, shift)
        if start < 0 or end <= start:
            continue
        try:
            reads = bam.fetch(str(row.genome), start, end)
        except ValueError:
            continue
        for read in reads:
            if read.is_unmapped or read.reference_start < start or read.reference_start >= end:
                continue
            if paired_only and read.mate_is_unmapped:
                continue
            forward = not read.is_reverse
            is_five = forward if row.strand == "+" else not forward
            if is_five:
                n5 += 1
            else:
                n3 += 1
    return n5, n3


def _interval(tss: int, strand: str, offset: int, window: int, upstream: int) -> tuple[int, int]:
    """0-based half-open genomic interval for one oriented window."""
    if strand == "+":
        first = tss - upstream + offset
        return first - 1, first - 1 + window
    last = tss + upstream - offset
    first = last - window + 1
    return first - 1, last


def _tests(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    paired0 = frame.loc[(frame["mode"] == "paired") & (frame["offset"] == 0)].set_index(["sample", "species"])
    paired100 = frame.loc[(frame["mode"] == "paired") & (frame["offset"] == 100)].set_index(["sample", "species"])
    single0 = frame.loc[(frame["mode"] == "single") & (frame["offset"] == 0)].set_index(["sample", "species"])
    shared = paired0.index.intersection(paired100.index)
    if len(shared) >= 2:
        result = ttest_rel(paired0.loc[shared, "stabilised_ratio"], paired100.loc[shared, "stabilised_ratio"])
        rows.append({"contrast": "paired_offset0_vs_100", "n": int(len(shared)), "pvalue": float(result.pvalue)})
    shared = paired0.index.intersection(single0.index)
    if len(shared) >= 2:
        result = ttest_rel(paired0.loc[shared, "stabilised_ratio"], single0.loc[shared, "stabilised_ratio"])
        rows.append({"contrast": "paired_vs_single_offset0", "n": int(len(shared)), "pvalue": float(result.pvalue)})
    if not rows:
        result = ttest_ind([0], [0])
        rows.append({"contrast": "insufficient_pairs", "n": 0, "pvalue": float("nan")})
    return pd.DataFrame(rows)
