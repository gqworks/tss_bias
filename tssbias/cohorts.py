"""Sample manifests, species pins, and MetaPhlAn abundance filters.

Relative abundance is a percent. MetaPhlAn species profiles are used as
published (already percentages). Long-read tables shipped under ``src/`` are
clade count tables; species rows are converted to a percentage of the
species-level column sum before the 0.5% rule is applied.

Species tables are MetaPhlAn 4 output written by
``python -m tssbias metaphlan <manifest>`` to
``{data_root}/mpa4_res/<cohort>/merged_mpa.txt``. This loader reads that
file. A count table whose column sum exceeds 100 is converted to a
percentage of the species-level column sum before the 0.5% rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from tssbias.errors import MissingInputError, require_paths


@dataclass(frozen=True)
class Sample:
    """One sequencing sample.

    ``rna_id`` is set for cohorts with matched metatranscriptomes (iHMP).
    ``longread_id`` is set for the matched short/long cohort.
    """

    sample_id: str
    rna_id: str | None = None
    longread_id: str | None = None


def read_manifest(path: Path) -> list[Sample]:
    """Read an ID-only manifest.

    Inputs:
        path: TSV with header ``sample_id`` and optional ``rna_id``,
            ``illumina_srr``, ``longread_srr``.
    Outputs:
        Samples in file order. For matched manifests, ``sample_id`` is the
        Illumina SRR when that column is present.
    """
    if not path.exists():
        raise MissingInputError(f"sample manifest for {path.name}", [path])
    frame = pd.read_csv(path, sep="\t", dtype=str)
    if "illumina_srr" in frame.columns:
        samples = []
        for row in frame.itertuples(index=False):
            samples.append(
                Sample(
                    sample_id=str(row.illumina_srr),
                    longread_id=str(row.longread_srr) if hasattr(row, "longread_srr") else None,
                )
            )
        return samples
    if "sample_id" not in frame.columns:
        raise ValueError(f"{path} needs a sample_id column")
    samples = []
    for row in frame.itertuples(index=False):
        rna = getattr(row, "rna_id", None)
        longread = getattr(row, "longread_srr", None)
        samples.append(
            Sample(
                sample_id=str(row.sample_id),
                rna_id=None if rna is None or str(rna) == "nan" else str(rna),
                longread_id=None if longread is None or str(longread) == "nan" else str(longread),
            )
        )
    return samples


def _species_key(clade: str) -> str | None:
    if "t__" in clade.split("|")[-1]:
        return None
    token = clade.split("|")[-1]
    if token.startswith("s__"):
        return token
    return None


def read_abundance(path: Path) -> pd.DataFrame:
    """Read a species-by-sample relative-abundance table in percent.

    Inputs:
        path: MetaPhlAn merged profile or a count table with a clade column.
    Outputs:
        DataFrame indexed by ``s__`` species name, columns are samples,
        values are percentages. Strain rows (``t__``) are dropped.
    """
    if not path.exists():
        raise MissingInputError("species abundance table", [path])
    frame = pd.read_csv(path, sep="\t", comment="#")
    clade_col = "clade_name" if "clade_name" in frame.columns else frame.columns[0]
    frame = frame.rename(columns={clade_col: "clade_name"})
    if "NCBI_tax_id" in frame.columns:
        frame = frame.drop(columns=["NCBI_tax_id"])
    keys = frame["clade_name"].astype(str).map(_species_key)
    frame = frame.loc[keys.notna()].copy()
    frame.index = keys[keys.notna()].to_numpy()
    frame = frame.drop(columns=["clade_name"])
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.groupby(level=0).sum()
    # Count tables (long-read Kraken/MPA dumps) have column sums far above 100.
    column_max = float(frame.max(numeric_only=True).max())
    if column_max > 100:
        totals = frame.sum(axis=0).replace(0, pd.NA)
        frame = 100.0 * frame.divide(totals, axis=1)
    return frame


def abundant_pairs(
    abundance: pd.DataFrame,
    samples: list[Sample],
    species_names: list[str],
    min_pct: float,
    min_samples: int,
) -> pd.DataFrame:
    """Apply the manuscript abundance screen.

    Inputs:
        abundance: species x sample percentages from :func:`read_abundance`.
        samples: cohort manifest, used for column order and membership.
        species_names: pinned species for this analysis.
        min_pct: minimum relative abundance (0.5).
        min_samples: minimum number of samples at or above ``min_pct``.
    Outputs:
        Boolean frame (species x sample) True where that pair is analysed.
        Species that fail the sample-count rule are absent. Samples missing
        from the abundance table raise MissingInputError.
    """
    sample_ids = [sample.sample_id for sample in samples]
    missing_samples = [sample for sample in sample_ids if sample not in abundance.columns]
    if missing_samples:
        preview = ", ".join(missing_samples[:8])
        extra = "" if len(missing_samples) <= 8 else f" (+{len(missing_samples) - 8} more)"
        raise MissingInputError(
            f"abundance table is missing {len(missing_samples)} manifest samples: {preview}{extra}",
            [],
        )
    present = [name for name in species_names if name in abundance.index]
    missing_species = [name for name in species_names if name not in abundance.index]
    if not present:
        raise MissingInputError(
            "none of the pinned species are in the abundance table: " + ", ".join(species_names),
            [],
        )
    if missing_species:
        # Pinned species that are simply absent from this cohort are not a
        # missing file. They fail the abundance rule and are skipped.
        pass
    subset = abundance.loc[present, sample_ids]
    above = subset >= min_pct
    keep = above.sum(axis=1) >= min_samples
    return above.loc[keep]


def fastq_paths(fastq_dir: Path, sample_id: str, seq_type: str) -> list[Path]:
    """Expected read files for one sample.

    Inputs:
        fastq_dir: cohort directory from config.
        sample_id: manifest id.
        seq_type: ``illumina`` (paired ``_1/_2``), or ``pacbio`` / ``nanopore``
            (single ``{id}.fastq.gz``).
    Outputs:
        Paths in read order. Existence is not checked here.
    """
    if seq_type == "illumina":
        return [
            fastq_dir / f"{sample_id}_1.fastq.gz",
            fastq_dir / f"{sample_id}_2.fastq.gz",
        ]
    return [fastq_dir / f"{sample_id}.fastq.gz"]


def longread_fastq(directory: Path, srr: str) -> Path | None:
    """FASTQ written by ``download`` for one long-read accession.

    A paired dump is ``{srr}_1.fastq.gz`` and ``_2``. A single-end dump is
    ``{srr}.fastq.gz``, or ``{srr}_1.fastq.gz`` when that is the only file
    ``fasterq-dump`` emitted. Returns the file to align, or None when neither
    single-end name is present.
    """
    single = directory / f"{srr}.fastq.gz"
    if single.exists():
        return single
    mate1 = directory / f"{srr}_1.fastq.gz"
    mate2 = directory / f"{srr}_2.fastq.gz"
    if mate1.exists() and not mate2.exists():
        return mate1
    return None


def require_existing(what: str, paths: list[Path]) -> None:
    """Wrapper so commands share one missing-input message."""
    require_paths(what, paths)
