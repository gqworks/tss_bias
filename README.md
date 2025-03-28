# tss_bias

Scripts and source files to guide the reproducibility of results from our manuscript: "Metagenomic coverage bias at transcription start sites is correlated with gene expression"

Each python script is responsible for a specific analysis performed in the manuscript, except for calc_coverage.py which is required to be run at the beginning to collect relevant species genome references, perform read alignment and coverage calculation. In order to run this script, the raw metagenomic sequencing data needs to be avaiable which needs to be seperately aquired through NCBI. Due to the high computational resource requirements and large intermediate files produced from this step, we have made speciesXsample TSS coverage matrices available on https://figshare.com/projects/Metagenomic_coverage_bias_at_transcription_start_sites_is_correlated_with_gene_expression/242600.

