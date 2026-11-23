"""Prodigal GFF parsing and the manuscript TSS filter.

A TSS is the gene start: GFF start on ``+`` and GFF end on ``-``. A TSS is
kept when it is more than ``window_bp`` (500) bases away from the neighbouring
gene starts on the same contig. Gene bodies may still overlap that flank; only
neighbouring starts are excluded. Contig-edge windows are dropped later, when
coverage is extracted.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def parse_prodigal_gff(path: Path) -> pd.DataFrame:
    """Parse a Prodigal GFF into gene rows.

    Inputs:
        path: GFF written by ``prodigal -f gff``.
    Outputs:
        DataFrame with columns genome, gene_left, gene_right, strand, desc,
        tss, locus_tag, gene_length. ``tss`` is 1-based.
    """
    rows: list[dict] = []
    for line in Path(path).read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 9 or parts[2] not in {"CDS", "gene"}:
            continue
        start, end = int(parts[3]), int(parts[4])
        strand = parts[6]
        desc = parts[8]
        locus = _locus_tag(desc)
        tss = start if strand == "+" else end
        rows.append(
            {
                "genome": parts[0],
                "gene_left": start,
                "gene_right": end,
                "strand": strand,
                "desc": desc,
                "tss": tss,
                "locus_tag": locus,
                "gene_length": abs(end - start) + 1,
            }
        )
    if not rows:
        raise ValueError(f"no CDS records in {path}")
    return pd.DataFrame(rows)


def _locus_tag(desc: str) -> str:
    for field in desc.split(";"):
        if field.startswith("ID="):
            return field[3:]
    return desc.split(";")[0]


def filter_isolated_tss(genes: pd.DataFrame, window_bp: int = 500) -> pd.DataFrame:
    """Keep gene starts more than ``window_bp`` from neighbouring starts.

    Inputs:
        genes: output of :func:`parse_prodigal_gff`.
        window_bp: minimum exclusive distance to the previous and next start.
    Outputs:
        Filtered frame, sorted by genome and TSS. A contig with one gene is kept.
    """
    kept: list[pd.Series] = []
    ordered = genes.sort_values(["genome", "tss", "locus_tag"]).reset_index(drop=True)
    for _, subset in ordered.groupby("genome", sort=False):
        subset = subset.reset_index(drop=True)
        n = len(subset)
        if n == 1:
            kept.append(subset.iloc[0])
            continue
        for i in range(n):
            tss = float(subset.at[i, "tss"])
            if i > 0 and not (tss - window_bp > float(subset.at[i - 1, "tss"])):
                continue
            if i < n - 1 and not (tss + window_bp < float(subset.at[i + 1, "tss"])):
                continue
            kept.append(subset.iloc[i])
    if not kept:
        return ordered.iloc[0:0].copy()
    return pd.DataFrame(kept).reset_index(drop=True)


def read_tss_table(path: Path) -> pd.DataFrame:
    """Read a filtered TSS table written by prepare-references.

    Inputs:
        path: CSV with the columns from :func:`filter_isolated_tss`.
    Outputs:
        DataFrame. Raises FileNotFoundError if ``path`` is missing; callers
        that speak to users should catch that and raise MissingInputError.
    """
    frame = pd.read_csv(path)
    required = {"genome", "strand", "tss", "locus_tag", "gene_length"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    return frame
