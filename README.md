# tss_bias

Scripts and package layout for reproducing results from:

**Metagenomic coverage bias at transcription start sites is correlated with gene expression**

This branch restructures the lab notebook into a **config-driven, NCI Gadi–oriented** layout without changing the scientific methods encoded in the analysis scripts.

## Quick start (Gadi)

1. Copy and edit config: `cp config/gadi.example.yaml config/gadi.local.yaml`
2. Set `gadi.project`, `paths.data_root`, `paths.scratch_root`
3. `export TSS_CONFIG=$PWD/config/gadi.local.yaml TSS_REPO=$PWD`
4. Create env: `conda env create -f environment.yml && conda activate tss_bias`
5. Stage data per [docs/DATA_STAGING.md](docs/DATA_STAGING.md)
6. Submit jobs under [jobs/gadi/](jobs/gadi/) — see [docs/GADI.md](docs/GADI.md)

```bash
python calc_coverage.py --cohort hmp2          # primary Illumina discovery matrices
python plot_heatmap_scatter.py --stage discovery
```

A naive `python calc_coverage.py` **without** `--cohort` exits with guidance (it no longer silently runs only long-read). Primary Illumina analyses are restored behind explicit CLI flags.

## Repository layout

| Path | Purpose |
| --- | --- |
| `tss_bias/` | Python package (config, paths, species pins, coverage, bias helpers) |
| `config/` | `default.yaml`, `gadi.example.yaml`, `species_accessions.yaml` |
| `data/sample_lists/` | Sample IDs only (no `/groups/cgsd/...` roots) |
| `data/reference/` | Bundled long-read MPA tables, Promotech example CSV |
| `jobs/gadi/` | PBS Pro templates (`ncpus`/`mem`/`storage`/`-P`) |
| `scripts/gc_adjusted_lmm.py` | **Stub** documenting missing GC-adjusted LMM |
| `calc_coverage.py`, `plot_*.py`, … | Entry scripts (legacy names kept; bootstrapped via config) |
| `metaphlan3_res/` | Partial MetaPhlAn3 fragments — **not** a drop-in for MPA4 |
| `mock_community_analysis/` | Mock / matched short–long analyses |

## Figshare matrices

Species×sample TSS coverage matrices (plot shortcut):  
https://figshare.com/projects/Metagenomic_coverage_bias_at_transcription_start_sites_is_correlated_with_gene_expression/242600  

Raw metagenomes must still be obtained from NCBI/ENA (or local archives for ARG/HYP). See [docs/DATA_STAGING.md](docs/DATA_STAGING.md) and [docs/FIGURE_MAP.md](docs/FIGURE_MAP.md).

## Important gaps

- **`mpa4_res/merged_mpa.txt` is not in git** — rebuild with MetaPhlAn4 (jobs provided).
- **GC-adjusted LMM** claimed in the manuscript is **not implemented** here (stub only).
- Promotech, Kraken DB, CheckM2, ART require local installs/modules on Gadi.
- Methods text historically cited MetaPhlAn3; code expects MetaPhlAn4 tables.
