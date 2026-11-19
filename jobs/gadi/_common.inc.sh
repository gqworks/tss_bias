# Shared Gadi preamble — sourced by job scripts after PBS directives.
# Expects: TSS_REPO (clone path), TSS_CONFIG (yaml), optional conda env name.

set -euo pipefail

: "${TSS_REPO:?Set TSS_REPO to the cloned tss_bias repository root}"
: "${TSS_CONFIG:=${TSS_REPO}/config/gadi.local.yaml}"

cd "${TSS_REPO}"
export PYTHONPATH="${TSS_REPO}${PYTHONPATH:+:$PYTHONPATH}"
export TSS_CONFIG

if [[ -n "${TSS_CONDA_ENV:-}" ]]; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${TSS_CONDA_ENV}"
fi

# Typical Gadi bioinformatics modules (adjust names to current NCI module list)
module load python3/3.11.0 2>/dev/null || true
module load bowtie2 2>/dev/null || true
module load bwa 2>/dev/null || true
module load samtools 2>/dev/null || true
module load parallel 2>/dev/null || true

echo "host=$(hostname)  ncpus=${PBS_NCPUS:-?}  jobid=${PBS_JOBID:-local}"
echo "TSS_REPO=${TSS_REPO}"
echo "TSS_CONFIG=${TSS_CONFIG}"
