#!/bin/bash
# Thin local/Gadi launcher. Prefer jobs/gadi/*.pbs on the cluster.
# Usage:
#   export TSS_CONFIG=config/gadi.local.yaml
#   ./main.sh discovery
#   ./main.sh coverage-hmp2
#   ./main.sh all-illumina
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export TSS_REPO="${TSS_REPO:-$ROOT}"
export PYTHONPATH="${TSS_REPO}${PYTHONPATH:+:$PYTHONPATH}"
cd "$TSS_REPO"

cmd="${1:-help}"
case "$cmd" in
  discovery)
    python plot_heatmap_scatter.py --stage discovery
    ;;
  coverage-hmp2)
    python calc_coverage.py --cohort hmp2
    ;;
  all-illumina)
    python calc_coverage.py --all-illumina
    ;;
  longread)
    python calc_coverage.py --cohort longread
    python plot_heatmap_scatter.py --stage longread
    ;;
  smoke)
    python calc_coverage.py --list-cohorts
    python scripts/gc_adjusted_lmm.py --write-schema /tmp/lmm_schema.csv || true
    ;;
  help|*)
    cat <<EOF
Usage: $0 {smoke|coverage-hmp2|all-illumina|discovery|longread}
See docs/GADI.md for PBS Pro submission.
EOF
    ;;
esac
