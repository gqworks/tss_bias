"""FASTA reader used by GC bias and reference preparation.

A small reader keeps the pipeline from importing Biopython on the statistics
path. Sequences are returned upper-cased without whitespace.
"""

from __future__ import annotations

from pathlib import Path


def read_fasta(path: Path) -> dict[str, str]:
    """Read a FASTA file into a contig -> sequence map.

    Inputs:
        path: nucleotide FASTA.
    Outputs:
        Dict keyed by the header token before the first whitespace.
    """
    sequences: dict[str, str] = {}
    name: str | None = None
    chunks: list[str] = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    sequences[name] = "".join(chunks).upper()
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if name is not None:
        sequences[name] = "".join(chunks).upper()
    if not sequences:
        raise ValueError(f"no sequences in {path}")
    return sequences


def contig_lengths(sequences: dict[str, str]) -> dict[str, int]:
    """Return contig lengths.

    Inputs:
        sequences: output of :func:`read_fasta`.
    Outputs:
        Dict of contig name to base count.
    """
    return {name: len(seq) for name, seq in sequences.items()}
