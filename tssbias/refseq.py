"""Resolve a MetaPhlAn species name to one RefSeq genome.

prepare-references queries NCBI for the reference genome, or the
representative genome when NCBI has not marked a reference. The result is
written to ``{refseq_dir}/resolved_accessions.tsv``. That file is a cache of
lookups, not a list of which species are analysed. The analysis set is the
MetaPhlAn abundance screen.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from tssbias.config import Species
from tssbias.errors import MissingInputError

REFERENCE = "reference genome"
REPRESENTATIVE = "representative genome"
COLUMNS = ["species", "taxid", "assembly", "ftp", "status"]


def resolved_table(refseq_dir: Path) -> Path:
    """Path of the accession cache written by prepare-references."""
    return refseq_dir / "resolved_accessions.tsv"


def binomial(metaphlan_name: str) -> str:
    """``s__Genus_species`` to ``Genus species``."""
    token = metaphlan_name[3:] if metaphlan_name.startswith("s__") else metaphlan_name
    return token.replace("_", " ")


def choose_assembly(records: list[dict]) -> dict | None:
    """Pick one RefSeq genome from NCBI assembly summaries.

    Reference genomes outrank representative genomes. Ties use the
    accession so the choice does not depend on return order. Records
    without a GCF accession or a RefSeq FTP path are ignored.
    """
    ranked: list[tuple[int, str, dict]] = []
    for record in records:
        category = str(record.get("refseq_category") or "").lower()
        ftp = str(record.get("ftppath_refseq") or "")
        accession = str(record.get("assemblyaccession") or "")
        if not accession.startswith("GCF_") or not ftp:
            continue
        if category == REFERENCE:
            rank = 0
        elif category == REPRESENTATIVE:
            rank = 1
        else:
            continue
        ranked.append((rank, accession, record))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1]))
    return ranked[0][2]


def assembly_from_summary(record: dict) -> tuple[str, str, int | None]:
    """Assembly directory name, https FTP directory, and taxid."""
    ftp = str(record["ftppath_refseq"])
    if ftp.startswith("ftp://"):
        ftp = "https://" + ftp[len("ftp://") :]
    ftp = ftp.rstrip("/")
    folder = ftp.rstrip("/").split("/")[-1]
    accession = str(record["assemblyaccession"])
    name = str(record.get("assemblyname") or "")
    assembly = folder if folder.startswith(accession) else f"{accession}_{name}"
    raw_taxid = record.get("taxid")
    taxid = None if raw_taxid in (None, "") else int(raw_taxid)
    return assembly, ftp, taxid


def lookup_refseq(metaphlan_name: str) -> Species:
    """Query NCBI Assembly. An empty assembly means no RefSeq genome.

    A network or HTTP failure raises MissingInputError. A successful query
    with no reference or representative genome returns an unresolved Species.
    """
    name = binomial(metaphlan_name)
    term = (
        f'"{name}"[Organism] AND latest_refseq[filter] AND '
        "(reference_genome[filter] OR representative_genome[filter])"
    )
    ids = _esearch(term)
    if not ids:
        return Species(metaphlan_name, "", "", None)
    records = _esummary(ids)
    chosen = choose_assembly(records)
    if chosen is None:
        return Species(metaphlan_name, "", "", None)
    assembly, ftp, taxid = assembly_from_summary(chosen)
    return Species(metaphlan_name, assembly, ftp, taxid)


def read_resolved(path: Path) -> dict[str, Species]:
    """Read the accession cache. Missing file returns an empty dict."""
    if not path.exists():
        return {}
    frame = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    species: dict[str, Species] = {}
    for row in frame.itertuples(index=False):
        taxid = str(row.taxid)
        species[str(row.species)] = Species(
            name=str(row.species),
            assembly=str(row.assembly),
            ftp=str(row.ftp),
            taxid=None if taxid in ("", "nan") else int(float(taxid)),
        )
    return species


def write_resolved(path: Path, species: dict[str, Species]) -> None:
    """Write the accession cache, sorted by species name."""
    rows = []
    for name in sorted(species):
        item = species[name]
        rows.append(
            {
                "species": item.name,
                "taxid": "" if item.taxid is None else str(item.taxid),
                "assembly": item.assembly,
                "ftp": item.ftp,
                "status": "resolved" if item.assembly else "unresolved",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, sep="\t", index=False)


def genomes_for(refseq_dir: Path, names: list[str]) -> list[Species]:
    """Resolved genomes for screened names.

    Names absent from the cache mean prepare-references has not been run
    for them. Names recorded as unresolved are skipped. If none of the
    names have a genome, raise MissingInputError.
    """
    path = resolved_table(refseq_dir)
    if not path.exists():
        raise MissingInputError(
            "RefSeq accessions are resolved by prepare-references. "
            "Run: python -m tssbias prepare-references --cohort <name>",
            [path],
        )
    known = read_resolved(path)
    missing = [name for name in names if name not in known]
    if missing:
        preview = ", ".join(missing[:8])
        extra = "" if len(missing) <= 8 else f" (+{len(missing) - 8} more)"
        raise MissingInputError(
            f"prepare-references has not recorded RefSeq accessions for {preview}{extra}",
            [path],
        )
    resolved = [known[name] for name in names if known[name].assembly]
    if not resolved:
        raise MissingInputError(
            "no screened species has a RefSeq reference or representative genome",
            [path],
        )
    return resolved


def _esearch(term: str) -> list[str]:
    query = urllib.parse.urlencode(
        {"db": "assembly", "term": term, "retmode": "json", "retmax": "20", "tool": "tssbias"}
    )
    payload = _get_json("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + query)
    return [str(item) for item in payload.get("esearchresult", {}).get("idlist", [])]


def _esummary(uids: list[str]) -> list[dict]:
    query = urllib.parse.urlencode(
        {"db": "assembly", "id": ",".join(uids), "retmode": "json", "tool": "tssbias"}
    )
    payload = _get_json("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?" + query)
    result = payload.get("result", {})
    records = []
    for uid in result.get("uids", uids):
        record = result.get(str(uid))
        if isinstance(record, dict):
            records.append(record)
    return records


def _get_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "tssbias"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except Exception as exc:
        raise MissingInputError(
            f"NCBI assembly query failed ({exc}). Retry prepare-references on a machine with network access.",
            [],
        ) from exc
