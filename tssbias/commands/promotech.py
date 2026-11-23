"""Promotech promoters versus Prodigal gene starts (Suppl. Fig. 1).

When predictions are absent, Promotech is run as
``promotech.py -pg -m RF-HOT`` and ``promotech.py -g -t 0.7 -m RF-HOT``.
The promoter coordinate is the prediction end on ``+`` and the start on ``-``.
The top N predictions by score are kept, with N equal to the number of
filtered Prodigal TSSs. Distance is to the nearest Prodigal TSS on the same
contig. Enrichment is the fraction of those promoters within 100 bp, tested
by rotating the Prodigal coordinates.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tssbias.commands.common import selected_species, write_table
from tssbias.config import Config
from tssbias.errors import MissingInputError, ToolError
from tssbias.external import run_command
from tssbias.logutil import get_logger
from tssbias.stats import greater_permutation_pvalue, rng_for, shift_tss_positions
from tssbias.tss import read_tss_table

LOG = get_logger("tssbias.promotech")


def run(cfg: Config, args: argparse.Namespace) -> None:
    """Write promoter-to-TSS distances and a 100 bp enrichment test.

    Outputs:
        ``results/promotech/{cohort}_distances.csv``
        ``results/promotech/{cohort}_enrichment.csv``
        ``results/promotech/{cohort}_distance_hist.png``
    """
    cohort = cfg.cohort(args.cohort)
    species = selected_species(cfg, cohort, args)
    distance_rows = []
    enrich_rows = []
    for item in species:
        fasta = cfg.fasta_path(item.assembly)
        tss_path = cfg.tss_table_path(item.assembly)
        if not fasta.exists() or not tss_path.exists():
            raise MissingInputError("Promotech needs the reference FASTA and filtered TSS table", [fasta, tss_path])
        predictions = cfg.promotech_dir / item.assembly / "genome_predictions.csv"
        if args.force or not predictions.exists():
            _run_promotech(cfg, fasta, predictions.parent)
        if not predictions.exists():
            raise MissingInputError("Promotech predictions were not produced", [predictions])
        tss = read_tss_table(tss_path)
        promoters = _load_promoters(predictions, len(tss))
        distances = _distances(promoters, tss)
        for distance, chrom in distances:
            distance_rows.append({"species": item.name, "assembly": item.assembly, "chrom": chrom, "distance_bp": distance})
        if distances:
            observed = float(np.mean([distance <= cfg.promotech_bp for distance, _chrom in distances]))
        else:
            observed = float("nan")
        pvalue, null_mean = _enrichment_pvalue(cfg, item.assembly, promoters, tss)
        enrich_rows.append(
            {
                "species": item.name,
                "assembly": item.assembly,
                "n_promoters": len(distances),
                "fraction_within_100bp": observed,
                "null_mean_fraction": null_mean,
                "pvalue": pvalue,
            }
        )
    out = cfg.results_dir / "promotech"
    distances_frame = pd.DataFrame(distance_rows)
    write_table(out / f"{cohort.name}_distances.csv", distances_frame)
    write_table(out / f"{cohort.name}_enrichment.csv", pd.DataFrame(enrich_rows))
    if not distances_frame.empty:
        fig, ax = plt.subplots()
        ax.hist(distances_frame["distance_bp"], bins=50, range=(0, 1000))
        ax.set_xlabel("Distance to closest Prodigal TSS (bp)")
        ax.set_ylabel("Promoters")
        ax.axvline(cfg.promotech_bp, color="red", linewidth=1)
        fig.tight_layout()
        fig.savefig(out / f"{cohort.name}_distance_hist.png", dpi=150)
        plt.close(fig)
    LOG.info("wrote Promotech distances for %s", cohort.name)


def _run_promotech(cfg: Config, fasta: Path, out_dir: Path) -> None:
    home = cfg.promotech_home or os.environ.get("PROMOTECH_HOME", "")
    script = Path(home) / "promotech.py" if home else None
    if script is None or not script.exists():
        raise ToolError(
            "promotech.py was not found. Set software.promotech_home or PROMOTECH_HOME "
            "to the Promotech checkout. Predictions can also be placed at "
            f"{out_dir / 'genome_predictions.csv'}."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    run_command(["python", str(script), "-pg", "-m", "RF-HOT", "-f", str(fasta), "-o", str(out_dir)])
    run_command(
        [
            "python",
            str(script),
            "-g",
            "-t",
            str(cfg.promotech_score_threshold),
            "-m",
            "RF-HOT",
            "-i",
            str(out_dir),
            "-o",
            str(out_dir),
        ]
    )


def _load_promoters(path: Path, n_keep: int) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    required = {"chrom", "start", "end", "score", "strand"}
    if not required <= set(frame.columns):
        raise MissingInputError(f"Promotech table needs columns {sorted(required)}", [path])
    frame = frame.sort_values("score", ascending=False).head(n_keep).copy()
    frame["promoter"] = np.where(frame["strand"] == "+", frame["end"], frame["start"]).astype(int)
    return frame


def _distances(promoters: pd.DataFrame, tss: pd.DataFrame) -> list[tuple[int, str]]:
    by_chrom: dict[str, np.ndarray] = {
        str(chrom): subset["tss"].to_numpy(dtype=int) for chrom, subset in tss.groupby("genome")
    }
    rows = []
    for row in promoters.itertuples(index=False):
        positions = by_chrom.get(str(row.chrom))
        if positions is None or positions.size == 0:
            continue
        rows.append((int(np.min(np.abs(positions - int(row.promoter)))), str(row.chrom)))
    return rows


def _enrichment_pvalue(cfg: Config, assembly: str, promoters: pd.DataFrame, tss: pd.DataFrame) -> tuple[float, float]:
    if promoters.empty:
        return float("nan"), float("nan")
    rng = rng_for(cfg.seed, "promotech", assembly)
    null = []
    for _ in range(cfg.n_permutations):
        shifted = tss.copy()
        pieces = []
        for genome, subset in tss.groupby("genome"):
            moved = shift_tss_positions(subset["tss"].to_numpy(), rng)
            block = subset.copy()
            block["tss"] = moved
            block["genome"] = genome
            pieces.append(block)
        shifted = pd.concat(pieces, ignore_index=True)
        distances = [distance for distance, _chrom in _distances(promoters, shifted)]
        if distances:
            null.append(float(np.mean([distance <= cfg.promotech_bp for distance in distances])))
    observed = _distances(promoters, tss)
    if not observed:
        return float("nan"), float("nan")
    obs = float(np.mean([distance <= cfg.promotech_bp for distance, _chrom in observed]))
    return greater_permutation_pvalue(obs, np.asarray(null, dtype=float)), float(np.mean(null) if null else float("nan"))
