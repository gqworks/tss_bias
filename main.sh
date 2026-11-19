#PBS -l nodes=1:ppn=16
#PBS -l mem=128gb
#PBS -l walltime=24:00:00
#PBS -q cgsd

cd $PBS_O_WORKDIR
source activate my_base
#python calc_coverage.py
#python plot_heatmap_scatter.py
#python plot_bias_vs_tpm.py
#python is_rna_analysis.py
#python upstream_unmapped_read_analysis.py
#python hetero_vs_tpm.py
#python assembled_ref_bias.py
#python relaxed_aligner.py
#python containment_hashing.py
python compare_promoter_prediction.py
