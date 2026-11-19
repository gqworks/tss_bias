#PBS -l nodes=1:ppn=64
#PBS -l mem=64gb
#PBS -l walltime=48:00:00
#PBS -q cgsd

cd $PBS_O_WORKDIR

python relaxed_aligner.py
