#!/bin/bash
# Submit one jobs/gadi/*.pbs script. The NCI project is not stored in the
# PBS file. Example:
#   export TSS_PROJECT=ab12
#   export TSS_CONFIG=$PWD/config/gadi.local.yaml
#   bash jobs/gadi/submit.sh jobs/gadi/discover.pbs
set -euo pipefail
: "${TSS_PROJECT:?Set TSS_PROJECT to the NCI project code}"
script=${1:?Pass a PBS script under jobs/gadi}
cd "$(dirname "$0")/../.."
qsub -P "${TSS_PROJECT}" \
  -l "storage=gdata/${TSS_PROJECT}+scratch/${TSS_PROJECT}" \
  -V \
  "$script"
