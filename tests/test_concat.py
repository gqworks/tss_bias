"""Concatenated-reference name handling. No aligner required."""

from __future__ import annotations

import pytest

from tssbias.concat import prefix_contig, rewrite_alignment, rewrite_sq, split_prefixed, write_concat_fasta
from tssbias.coverage import depth_from_table


def test_prefix_round_trip():
    assert prefix_contig("GCF_1", "NZ_1") == "GCF_1::NZ_1"
    assert split_prefixed("GCF_1::NZ_1") == ("GCF_1", "NZ_1")


def test_prefix_rejects_the_separator():
    with pytest.raises(ValueError):
        prefix_contig("GCF_1", "bad::name")


def test_rewrite_strips_contig_and_marks_a_cross_genome_mate():
    fields = ["r1", "99", "A::c1", "10", "60", "150M", "B::c2", "20", "100", "ACGT", "*"]
    out = rewrite_alignment(fields, "A")
    assert out is not None
    assert out[2] == "c1"
    assert out[6] == "*"
    assert out[7] == "0"
    assert int(out[1]) & 0x8


def test_rewrite_keeps_a_mate_on_the_same_genome():
    fields = ["r1", "99", "A::c1", "10", "60", "150M", "A::c2", "20", "100", "ACGT", "*"]
    out = rewrite_alignment(fields, "A")
    assert out is not None
    assert out[2] == "c1"
    assert out[6] == "c2"
    assert out[1] == "99"
    assert rewrite_alignment(fields, "B") is None


def test_sq_line_is_restored_per_assembly():
    line = "@SQ\tSN:ASM::c1\tLN:4\n"
    assert rewrite_sq(line, "ASM") == "@SQ\tSN:c1\tLN:4\n"
    assert rewrite_sq(line, "OTHER") is None


def test_concat_fasta_prefixes_headers(tmp_path):
    source = tmp_path / "a.fna"
    source.write_text(">c1 desc\nACGT\n>c2\nTT\n")
    dest = tmp_path / "out" / "concat.fna"
    write_concat_fasta([("ASM", source)], dest)
    assert dest.read_text() == ">ASM::c1\nACGT\n>ASM::c2\nTT\n"


def test_depth_table_keeps_zero_bases(tmp_path):
    path = tmp_path / "depth.tsv"
    path.write_text("c1\t1\t0\nc1\t2\t3\nc1\t4\t1\n")
    depth = depth_from_table(path)
    assert list(depth["c1"]) == [0.0, 3.0, 0.0, 1.0]
