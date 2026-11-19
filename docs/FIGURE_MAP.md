# Script ↔ analysis / figure map

Ad hoc `plots/` filenames are **not** manuscript figure numbers. Use this as a launch checklist.

| Theme | Script / stage | Typical outputs |
| --- | --- | --- |
| Coverage matrices | `calc_coverage.py --cohort …` | `matrices/`, `beds/`, `bams/`, `refseq_genomes/` |
| Discovery null + FDR + heatmaps | `plot_heatmap_scatter.py --stage discovery` | `plots/null_distributions/hmp2_*`, `plots/coverage_heatmaps/` |
| GC nulls | `--stage gc` | `plots/null_distributions/*_gc*` |
| ART mappability null | `--stage simu` | `*_simu*` under nulls/heatmaps |
| Long-read FDR | `--stage longread` | `longread_pvals_*` |
| Expression groups / GC vs bias | `plot_bias_vs_tpm.py` | `plots/gene_expression/` |
| Cohort correlation | `plot_bias_corr.py` | cohort corr heatmaps under `plots/` |
| Platform corr | `plot_bias_corr_platform.py` | Illumina vs PacBio/ONT |
| 5′ enrichment | `is_rna_analysis.py` | `plots/paired_end_analysis/` |
| Upstream mates | `upstream_unmapped_read_analysis.py` | ARG diversity plots |
| Hetero vs TPM | `hetero_vs_tpm.py` | needs `align` / calign |
| MAG remapping | `assembled_ref_bias.py` | often reads precomputed CSVs; assembly commented |
| Bowtie2 `--mp` | `relaxed_aligner.py` | `plots/relaxed_aligner/` |
| Promotech | `compare_promoter_prediction.py` | Suppl. Fig. 1 style |
| sylph | `containment_hashing.py` | Reviewer Fig. 1 style |
| Mock community | `mock_community_analysis/*` | mock bias CSVs |
| GC-adjusted LMM | `scripts/gc_adjusted_lmm.py` | **stub only — not implemented** |
