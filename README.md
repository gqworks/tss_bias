# TSS coverage bias

Re-executable analyses for "Metagenomic coverage bias at transcription start sites is correlated with gene expression".

The runnable path is `python -m tssbias`. The Python files at the repository root are the original lab scripts and are not the rerun interface. Configuration, seeds, and thresholds live in `config/default.yaml` and `config/species.yaml`. Copy `config/gadi.example.yaml` to `config/gadi.local.yaml` on NCI Gadi and point `TSS_CONFIG` at it. `TSS_PROJECT`, `TSS_DATA_ROOT`, `TSS_SCRATCH_ROOT`, `TSS_RESULTS_DIR`, and `TSS_SEED` override the file.

```bash
python -m tssbias check-inputs --cohort hmp2
python -m tssbias prepare-references --cohort hmp2
python -m tssbias discover --cohort hmp2
python -m tssbias permutation --cohort hmp2
python -m tssbias expression --cohort hmp2
python -m tssbias gc-lmm --cohort hmp2
```

A missing FASTQ, abundance table, BAM, or reference raises a missing-input error and writes nothing. `pytest` covers the metric definitions without those files.

## Definitions

* TSS: Prodigal gene start (GFF start on `+`, GFF end on `-`). Starts within 500 bp of a neighbouring start are dropped.
* Window: 1001 bp, 500 bp upstream, the TSS base, 500 bp downstream. Minus-strand windows are flipped so downstream is to the right.
* Scaling: each window is z-scored with the sample standard deviation (ddof=1).
* Coverage bias: mean scaled downstream flank minus mean scaled upstream flank. The TSS base is scaled with the window and is in neither mean. Zero-coverage windows are excluded.
* GC bias: GC% in the 500 bp downstream minus GC% in the 500 bp upstream.
* Abundance-normalised TPM: `norm_tpm = 1e6 * kallisto_tpm / mapped_metagenomic_reads`. Groups on `norm_tpm`: zero (`== 0`), low (`< 100`), mid (`< 1000`), high (`>= 1000`).
* Log expression: `ln(norm_tpm + 0.001)`.
* Permutation: rotate TSS coordinates on each contig so successive gaps stay the same (modulus = the last TSS). One-sided p = `min(1, (#{null mean > observed} + 1) / n)`. Illumina n = 500, long-read n = 100. Benjamini–Hochberg FDR 0.05 across the tests in the run.
* Species must be at or above 0.5% relative abundance in enough samples (iHMP 30, Franzosa 5, ARG 10, HYP 5, PacBio/Nanopore 4).

The lab scripts summed an asymmetric 500-vs-501 bp split of the z-scored window. This pipeline uses the manuscript mean of the two 500 bp flanks. Accessions follow the assemblies those scripts actually ran. A methods table in the revision draft swaps *F. prausnitzii* and *P. vulgatus*; `config/species.yaml` does not.

The manuscript says MetaPhlAn3. Later scripts read a merged species table at `mpa4_res/merged_mpa.txt`. `metaphlan` runs the `metaphlan` binary on PATH and writes that path. The partial `metaphlan3_res/` checkout is not that table. Long-read tables in `src/` are counts and are converted to a percentage of the species-level column sum before the 0.5% rule.

## GC-adjusted mixed model

Manuscript formula, fit by `gc-lmm`:

```text
scale(log(norm_tpm + 0.001)) ~ scale(coverage_bias) + scale(gc_bias) + (1 | sample) + (1 | species)
```

`scale` is a z-score (ddof=1) on the rows that enter the model. The outcome is log abundance-normalised expression. Fixed effects are coverage bias and GC bias. Random intercepts are sample and species. TSSs with no coverage in the 1001 bp window are dropped. The fit is REML (`statsmodels` MixedLM). The manuscript used lmerTest, which reports Satterthwaite degrees of freedom; this rerun reports Wald z tests for the same formula. Coefficients are written under `results/lmm/`.

The upstream-mate model (`upstream-scores`) is the second documented mixed model:

```text
scale(coverage_bias) ~ scale(upstream_alignment_score) + (1 | sample) + (1 | species)
```

Scores are median semi-global alignments (`align.calign`, DNAFULL) of the unmapped-or-paired 5' mate against the 500 bp upstream flank.

## Figure to command

| Manuscript piece | Command |
| --- | --- |
| Primary Illumina discovery matrices (Fig. 1 coverage) | `python -m tssbias discover --cohort hmp2` |
| Position-permutation null and BH FDR | `python -m tssbias permutation --cohort hmp2` |
| GC-bias permutation | `python -m tssbias permutation --cohort hmp2 --metric gc` |
| Expression groups and abundance-normalised TPM | `python -m tssbias expression --cohort hmp2` |
| GC-adjusted linear mixed model | `python -m tssbias gc-lmm --cohort hmp2` |
| Cohort replication (iHMP, Franzosa, HYP, ARG) | `python -m tssbias cohort-correlation` |
| Illumina vs PacBio vs Nanopore | `python -m tssbias platform-comparison` |
| Long-read test restricted to genes > 1 kb | `python -m tssbias permutation --cohort pacbio --min-gene-bp 1000` |
| ART mappability control | `python -m tssbias mappability --cohort hmp2` |
| Mock community (ATCC MSA-1003) | `python -m tssbias mock` |
| Short reads on same-sample long reads | `python -m tssbias matched-longread` |
| 5' vs 3' enrichment | `python -m tssbias five-prime --cohort hmp2` |
| Upstream alignment scores | `python -m tssbias upstream-scores --cohort arg` |
| MAG remapping (Suppl. Fig. 5) | `python -m tssbias mag-bias --cohort hmp2` |
| Bowtie2 mismatch penalty (Suppl. Fig. 7) | `python -m tssbias mismatch-penalty --cohort hmp2` |
| Substitutions vs expression (Suppl. Fig. 8) | `python -m tssbias substitutions --cohort hmp2` |
| Promotech vs Prodigal (Suppl. Fig. 1) | `python -m tssbias promotech --cohort hmp2` |
| sylph containment ANI | `python -m tssbias sylph --cohort hmp2` |
| Rebuild the abundance table | `python -m tssbias metaphlan --cohort hmp2` |

Franzosa replication uses the 8 RNAlater accessions in `data/manifests/franzosa_rnalater.tsv`. iHMP is the 78 pairs in `data/manifests/hmp2.tsv` (DNA id, RNA id), BioProject PRJNA389280. Array jobs pass `--sample-index` (0-based).

## What runs here, and what needs Gadi

Without staged reads the test suite, `--help`, and `check-inputs` run. Every analysis command exits with a missing-input message until its files exist. Nothing in this pipeline fabricates coverage, p-values, or model coefficients.

A full rerun needs, on Gadi or another machine:

* NCI project code (`TSS_PROJECT`) and `config/gadi.local.yaml` (`/g/data/{project}` for references and results, `/scratch/{project}` for BAMs and bigWigs).
* FASTQs named `{sample}_1.fastq.gz` and `{sample}_2.fastq.gz` (Illumina) or `{sample}.fastq.gz` (PacBio, Nanopore) under each cohort's `fastq_dir`.
* `mpa4_res/merged_mpa.txt` from `metaphlan` (the methods text says MetaPhlAn3; pin the binary you intend).
* Prodigal 2.6.3, Bowtie2, BWA, samtools, deepTools `bamCoverage`, kallisto, and the extra tools named in `environment.yml` for ART, MEGAHIT, MetaBAT2, Kraken2, CheckM2, sylph, minimap2, and Promotech.
* Kraken2 DB (`software.kraken_db`) for MAG bin selection. Promotech checkout (`PROMOTECH_HOME`). The `align` package (`align.calign`, DNAFULL) for upstream scores.
* Mock community: the 20 ATCC genomes and `SRR8359173` reads concatenated/staged as `check-inputs` does not cover `mock`; the command prints the expected paths. HiFi accession in the original download script is `SRR11606871`.

Submit a PBS script with the project and storage directive supplied at `qsub` time:

```bash
export TSS_PROJECT=ab12
export TSS_CONFIG=$PWD/config/gadi.local.yaml
bash jobs/gadi/submit.sh jobs/gadi/discover.pbs
```

Figshare project 242600 holds the original headerless matrices. This pipeline writes a headed CSV (`locus_tag`, `pos_-500` … `pos_500`) and does not read those files in place.

## Environment

Python dependencies are pinned in `requirements.txt` and `pyproject.toml` (numpy, pandas, scipy, statsmodels, matplotlib, PyYAML). `environment.yml` adds the bioconda tools. External programs are resolved from `PATH` at runtime.
