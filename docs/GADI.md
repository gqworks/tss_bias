# Running on NCI Gadi

## 1. Clone and configure

```bash
git clone https://github.com/gqworks/tss_bias.git
cd tss_bias
git checkout cursor/refactor-gadi-ready-d7fb   # or main once merged

cp config/gadi.example.yaml config/gadi.local.yaml
# Edit: gadi.project, paths.data_root, paths.scratch_root, software.* DBs
export TSS_CONFIG=$PWD/config/gadi.local.yaml
export TSS_REPO=$PWD
export TSS_DATA_ROOT=/g/data/$PROJECT/tss_bias
export TSS_SCRATCH_ROOT=/scratch/$PROJECT/tss_bias
```

Replace every `REPLACE_NCI_PROJECT` in `jobs/gadi/*.pbs` with your project code (or sed at submit time).

## 2. Software environment

**Option A — conda (recommended for Python + many tools):**

```bash
module load python3/3.11.0   # if required on Gadi before conda
conda env create -f environment.yml
conda activate tss_bias
# or: pip install -r requirements.txt && pip install -e .
```

**Option B — Gadi modules + pip:** install Python deps from `requirements.txt`; load aligner modules in job scripts (`bowtie2`, `bwa`, `samtools`). Confirm current module names with `module avail` — they change.

| Tool | Used by | Pin / note |
| --- | --- | --- |
| bowtie2, bwa, samtools | `calc_coverage`, relaxed, MAG | conda or modules |
| prodigal, deepTools | coverage matrices | conda (`deeptools`, `prodigal`) |
| MetaPhlAn4 | abundance tables | **not** bundled DB; set `$METAPHLAN_DB` if needed |
| kallisto | MTX TPM | conda |
| ART | simu nulls | module/name varies (`software.art_module`) |
| MEGAHIT, MetaBat2, Kraken2, CheckM2 | MAG path | optional; Kraken DB via config |
| sylph | containment | optional |
| Promotech | promoter validation | external install + `$PROMOTECH_HOME` |

## 3. Stage data

Follow [DATA_STAGING.md](DATA_STAGING.md). Minimum for discovery plots:

1. HMP2 FASTQs under `cohorts.hmp2.fastq_dir`
2. `mpa4_res/merged_mpa.txt`
3. Either run `calc_coverage.py --cohort hmp2` **or** unpack Figshare matrices (+ beds if running nulls)

## 4. Submit jobs

```bash
mkdir -p logs
# smoke (no FASTQs required)
qsub -v TSS_REPO=$PWD,TSS_CONFIG=$TSS_CONFIG jobs/gadi/00_smoke_test.pbs

# MetaPhlAn4 array (prepare DNA ID list first)
awk -F'\t' '!/^#/ && NF{print $1}' data/sample_lists/hmp2_samples > /scratch/$PROJECT/hmp2_dna_ids.txt
N=$(wc -l < /scratch/$PROJECT/hmp2_dna_ids.txt)
qsub -J 1-${N} -v TSS_REPO=$PWD,SAMPLE_LIST=/scratch/$PROJECT/hmp2_dna_ids.txt,FASTQ_DIR=$TSS_DATA_ROOT/fastq/hmp2 \
  jobs/gadi/03_metaphlan_array.pbs
bash jobs/gadi/04_merge_metaphlan.sh

# Coverage + discovery figures
qsub -v TSS_REPO=$PWD,TSS_CONFIG=$TSS_CONFIG,TSS_CONDA_ENV=tss_bias jobs/gadi/01_calc_coverage_hmp2.pbs
qsub -v TSS_REPO=$PWD,TSS_CONFIG=$TSS_CONFIG,TSS_CONDA_ENV=tss_bias jobs/gadi/02_plot_discovery.pbs
```

PBS Pro notes vs old Torque (`#PBS -q cgsd`, `nodes=1:ppn=`):

- Use `#PBS -l ncpus=,mem=,walltime=,storage=`
- `#PBS -P` project code
- `#PBS -l storage=gdata/PROJ+scratch/PROJ` so jobs can see both filesystems
- Prefer job arrays (`#PBS -J`) for per-sample MetaPhlAn / future per-sample×species coverage

Legacy Torque wrappers remain under `jobs/gadi/legacy/` for reference only.

## 5. CLI entry points (no more silent “only Promotech / only long-read”)

```bash
python calc_coverage.py --list-cohorts
python calc_coverage.py --cohort hmp2          # primary Illumina discovery
python calc_coverage.py --all-illumina
python calc_coverage.py --cohort longread
python plot_heatmap_scatter.py --stage discovery
python plot_heatmap_scatter.py --stage all
python scripts/gc_adjusted_lmm.py --help       # stub — see gap below
```

## 6. Known gaps Gordon should confirm

1. **NCI project code** and exact `gdata`/`scratch` directory layout  
2. **MetaPhlAn3 vs 4** — regenerate with 4 to match code, or reintroduce 3 tables deliberately  
3. **GC-adjusted LMM** — missing; stub only (`scripts/gc_adjusted_lmm.py`)  
4. **Promotech / Kraken / CheckM2 / ART** module or install paths on Gadi  
5. **BAM retention** — which revision analyses need scratch BAMs vs matrix-only  
6. **ARG/HYP FASTQ** provenance (local archives vs re-download)
