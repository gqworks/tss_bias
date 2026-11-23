"""Concatenate whole genomes for one sample and split the alignment back.

Contig names in the concatenated FASTA are ``{assembly}::{contig}``. The
split writes one BAM per assembly and restores the original contig name so
later commands still match the per-species FASTA and TSS table. References
are the full genomes, not TSS windows.
"""

from __future__ import annotations

from pathlib import Path

CONTIG_SEP = "::"


def prefix_contig(assembly: str, contig: str) -> str:
    """Contig name used inside the concatenated reference."""
    if CONTIG_SEP in contig or CONTIG_SEP in assembly:
        raise ValueError(f"contig or assembly name contains {CONTIG_SEP!r}: {assembly} {contig}")
    return f"{assembly}{CONTIG_SEP}{contig}"


def split_prefixed(name: str) -> tuple[str, str] | None:
    """Return ``(assembly, contig)`` or None when ``name`` is not prefixed."""
    if CONTIG_SEP not in name:
        return None
    assembly, contig = name.split(CONTIG_SEP, 1)
    if not assembly or not contig:
        return None
    return assembly, contig


def write_concat_fasta(fastas: list[tuple[str, Path]], destination: Path) -> None:
    """Write whole genomes, prefixing each contig with its assembly id.

    Inputs:
        fastas: ``(assembly, fasta path)`` in the order they should appear.
        destination: output FASTA. Parent directories are created.
    Outputs:
        FASTA at ``destination``. Sequence lines are copied unchanged.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    with partial.open("w") as out:
        for assembly, fasta in fastas:
            with fasta.open() as handle:
                for line in handle:
                    if line.startswith(">"):
                        token = line[1:].split()[0]
                        out.write(f">{prefix_contig(assembly, token)}\n")
                    elif line.strip():
                        out.write(line if line.endswith("\n") else line + "\n")
    partial.replace(destination)


def rewrite_alignment(fields: list[str], assembly: str) -> list[str] | None:
    """Restore original contig names on one SAM record.

    Inputs:
        fields: SAM columns, not including the newline.
        assembly: assembly this BAM is being split into.
    Outputs:
        Updated columns, or None when the read is not aligned to ``assembly``.
        A mate on a different assembly is marked unmapped.
    """
    if len(fields) < 7 or fields[0].startswith("@"):
        return None
    parsed = split_prefixed(fields[2])
    if parsed is None or parsed[0] != assembly:
        return None
    updated = list(fields)
    updated[2] = parsed[1]
    mate = split_prefixed(fields[6]) if fields[6] not in ("*", "=") else None
    if mate is not None and mate[0] != assembly:
        flag = int(updated[1]) | 0x8
        updated[1] = str(flag)
        updated[6] = "*"
        updated[7] = "0"
        updated[8] = "0"
    elif mate is not None:
        updated[6] = mate[1]
    return updated


def rewrite_sq(line: str, assembly: str) -> str | None:
    """Rewrite one ``@SQ`` line, or return None if it is another assembly."""
    if not line.startswith("@SQ"):
        return None
    parts = line.rstrip("\n").split("\t")
    name = None
    for part in parts:
        if part.startswith("SN:"):
            name = part[3:]
            break
    parsed = split_prefixed(name or "")
    if parsed is None or parsed[0] != assembly:
        return None
    rewritten = []
    for part in parts:
        if part.startswith("SN:"):
            rewritten.append(f"SN:{parsed[1]}")
        else:
            rewritten.append(part)
    return "\t".join(rewritten) + "\n"
