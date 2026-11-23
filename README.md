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
* Species must be at or above 0.5% relative abundance in enough samples (iHMP 30, Franzosa 5, Liu et al. 10, HMP Phase III stool 5, PacBio/Nanopore 4).

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
| Cohort replication (iHMP, Franzosa, HMP Phase III stool, Liu et al.) | `python -m tssbias cohort-correlation` |
| HMP Phase III stool discovery (public replacement for HYP) | `python -m tssbias discover --cohort hmp3_stool` |
| Liu et al. discovery (public replacement for ARG) | `python -m tssbias discover --cohort liu2016` |
| Illumina vs PacBio vs Nanopore | `python -m tssbias platform-comparison` |
| Long-read test restricted to genes > 1 kb | `python -m tssbias permutation --cohort pacbio --min-gene-bp 1000` |
| ART mappability control | `python -m tssbias mappability --cohort hmp2` |
| Mock community (ATCC MSA-1003) | `python -m tssbias mock` |
| Short reads on same-sample long reads | `python -m tssbias matched-longread` |
| 5' vs 3' enrichment | `python -m tssbias five-prime --cohort hmp2` |
| Upstream alignment scores | `python -m tssbias upstream-scores --cohort liu2016` |
| MAG remapping (Suppl. Fig. 5) | `python -m tssbias mag-bias --cohort hmp2` |
| Bowtie2 mismatch penalty (Suppl. Fig. 7) | `python -m tssbias mismatch-penalty --cohort hmp2` |
| Substitutions vs expression (Suppl. Fig. 8) | `python -m tssbias substitutions --cohort hmp2` |
| Promotech vs Prodigal (Suppl. Fig. 1) | `python -m tssbias promotech --cohort hmp2` |
| sylph containment ANI | `python -m tssbias sylph --cohort hmp2` |
| Rebuild the abundance table | `python -m tssbias metaphlan --cohort hmp2` |

Franzosa replication uses the 8 RNAlater DNA runs in `data/manifests/franzosa_rnalater.tsv` (PRJNA188481). iHMP is the 78 DNA runs in `data/manifests/hmp2.tsv` (PRJNA389280), each with its paired RNA run in `rna_id`. Array jobs pass `--sample-index` (0-based). The column layout and the one-run rules are below.

## HMP Phase III stool (replaces HYP)

`hmp3_stool` is the public DNA shotgun cohort in place of the private HYP list. It is the same command slot: `discover`, `permutation`, `metaphlan`, and `cohort-correlation` take `--cohort hmp3_stool` (correlation always includes it). Abundance screen is the HYP bar, 0.5% in at least 5 samples, species set `correlation`. There is no paired RNA.

Source: BioProject PRJNA275349, SRA study SRP056641, ENA read-run table. Healthy US adult HMP Phase III WGS. The manifest is `data/manifests/hmp3_stool.tsv` (89 runs, 89 specimens). `sample_id` is the run accession. Every specimen that survives the rule is included.

Selection rule, applied in this order:

1. Keep `sample_alias` ending in `G_DNA_Stool` (stool; other body sites in this project are excluded).
2. Keep paired-end WGS metagenomes on Illumina HiSeq 2000 (`library_layout=PAIRED`, `library_strategy=WGS`, `library_source=METAGENOMIC`). All 96 stool runs match. The project's only single-end run is anterior nares.
3. Keep one run per alias. Seven aliases are deposited as two runs of the same BioSample. Keep the lexicographically smallest `run_accession`. That leaves 89 specimens. The dropped runs are SRR2175754, SRR2175794, SRR2240257, SRR2240287, SRR2240290, SRR2240318, and SRR2241210.
4. Do not collapse further. The alias is `{specimen}_zg_G_DNA_Stool`. `zg` is the stool-site token on every stool row (other sites use `zs1`, `zo1`, `zv2`, `zo7`, `zo9`). No specimen number is shared across aliases, so a repeat visit of the same donor is not visible. Public BioSample records have no subject id; that link is dbGaP phs000228 and is not used.
5. Sort the 89 aliases ascending. The manifest is that full list.

Stage paired FASTQs at `{data_root}/fastq/hmp3_stool/{SRR}_1.fastq.gz` and `{SRR}_2.fastq.gz`. The commands do not download this cohort. If a file is absent, `check-inputs` and `discover` exit 2 and write nothing. Rebuild abundance with `python -m tssbias metaphlan --cohort hmp3_stool` into `{data_root}/mpa4_res/hmp3_stool/merged_mpa.txt`. On Gadi, `TSS_COHORT=hmp3_stool` and a 1–89 array cover the manifest.

## Liu et al. (replaces ARG)

Display name: `Liu et al.` Internal id: `liu2016`. It uses the commands ARG used (`discover`, `permutation`, `metaphlan`, `upstream-scores`, and `cohort-correlation`) and the same abundance screen, 0.5% in at least 10 samples, on the species set `liu2016`. There is no paired RNA.

Source: Liu et al., Scientific Reports 2016, BioProject PRJNA328899, SRA study SRP080787. The paper describes 110 healthy Mongolian adults and TruSeq libraries with an average insert of 500 bp. ENA `nominal_length` is blank on every run. The archive instrument is Illumina HiSeq 4000 on all 110; the paper says HiSeq 2500. The manifest follows the archive instrument and does not invent a per-sample insert.

The manifest is `data/manifests/liu2016.tsv` (110 runs, 110 samples). Selection rule: keep every public run in that BioProject. Each `sample_alias` (`Sample_001` through `Sample_112`, with `Sample_052` and `Sample_101` absent from the archive) and each BioSample has exactly one run, so the one-run-per-sample rule drops nothing. Rows are sorted by `sample_alias`, then `run_accession`.

Stage paired FASTQs at `{data_root}/fastq/liu2016/{SRR}_1.fastq.gz` and `{SRR}_2.fastq.gz`. The commands do not download this cohort. If a file is absent, `check-inputs` and `discover` exit 2 and write nothing. Rebuild abundance with `python -m tssbias metaphlan --cohort liu2016` into `{data_root}/mpa4_res/liu2016/merged_mpa.txt`. On Gadi, `TSS_COHORT=liu2016` and a 1–110 array cover the manifest.

## iHMP (HMP2)

`data/manifests/hmp2.tsv` keeps the existing public DNA set: 78 runs, 78 samples, BioProject PRJNA389280, SRA study SRP115812. `sample_id` is the metagenomic WGS run. Columns match the other public manifests, plus `rna_id`, the paired metatranscriptomic run. DNA and RNA are different BioSamples, so the pair is the one already in this manifest, not a shared accession.

Selection rule: keep every DNA run already in the cohort. Each of those 78 aliases and BioSamples has exactly one run in the project, so the one-run-per-sample rule drops nothing. The other 300 metagenomes in PRJNA389280 are not added. Rows are sorted by `sample_alias`, then `sample_id`. No cluster path is stored.

## Franzosa RNAlater

`data/manifests/franzosa_rnalater.tsv` is the public Franzosa set, BioProject PRJNA188481, SRA study SRP019038. The public table has 80 runs and 32 aliases (8 people × RNAlater, ETOH, Whole, and Saliva).

Selection rule, applied in this order:

1. Keep stool stored in RNAlater. The person id is the `X########` prefix of `sample_alias`. Drop `ETOH` (ethanol), `Whole` (frozen), and `Saliva`.
2. Keep the paired-end WGS metagenome (`library_strategy=WGS`, `library_source=METAGENOMIC`). Each RNAlater alias also has two RNA-Seq runs; those are not this DNA cohort.
3. Keep one run per alias. Each RNAlater alias and BioSample has exactly one WGS run, so nothing further is dropped. That leaves 8 people, 8 runs.
4. Sort by `sample_alias`.

The previous 8-row list was already these eight accessions. The commands do not download the FASTQs.

## PacBio, Nanopore, and matched long reads

These three manifests use the same columns. `sample_id` is the SRA run. No cluster path is stored. Each alias and BioSample in the source list has one run, so the one-run-per-sample rule drops nothing. Rows are sorted by `sample_alias`, then `sample_id`.

`data/manifests/pacbio.tsv` is 11 Sequel II WGS runs, BioProject PRJNA754443, SRA study SRP332749. Layout is `SINGLE`. Aliases are `1_baseline` through `6_baseline` and the matching `*_treatment` rows.

`data/manifests/nanopore.tsv` is 10 PromethION WGS runs, BioProject PRJNA820119, SRA study SRP366030. Layout is `SINGLE`. Aliases are `CD91`–`CD100`.

`data/manifests/matched_longread.tsv` is the 10 Illumina NovaSeq runs from the same BioProject, one per donor. `sample_id` is that paired-end WGS run. `longread_srr` is the PromethION run whose alias is the same `CD` id (the Nanopore manifest). The short-read and long-read deposits are different BioSamples, so both accessions stay on the row. The commands do not download either FASTQ.

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
