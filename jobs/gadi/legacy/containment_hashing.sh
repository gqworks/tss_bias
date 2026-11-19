#PBS -l nodes=1:ppn=32
#PBS -l mem=64gb
#PBS -l walltime=24:00:00
#PBS -q cgsd

cd $PBS_O_WORKDIR

python containment_hashing.py
