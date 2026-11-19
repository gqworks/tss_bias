#!/bin/bash
# Merge per-sample MetaPhlAn tables into merged_mpa.txt (run on login node or short job).
set -euo pipefail
: "${TSS_REPO:?}"
MPA_OUT="${MPA_OUT:-${TSS_REPO}/workdir/mpa4_res}"
mkdir -p "${MPA_OUT}"
shopt -s nullglob
tables=("${MPA_OUT}"/*.txt)
# drop merged if present
filtered=()
for t in "${tables[@]}"; do
  [[ "$(basename "$t")" == "merged_mpa.txt" ]] && continue
  filtered+=("$t")
done
if [[ ${#filtered[@]} -eq 0 ]]; then
  echo "No per-sample tables in ${MPA_OUT}" >&2
  exit 1
fi
module load metaphlan 2>/dev/null || true
merge_metaphlan_tables.py "${filtered[@]}" > "${MPA_OUT}/merged_mpa.txt"
echo "Wrote ${MPA_OUT}/merged_mpa.txt (${#filtered[@]} samples)"
