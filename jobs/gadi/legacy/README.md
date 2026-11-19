# Legacy Torque / HKU (`#PBS -q cgsd`) scripts

These are retained for provenance. Do **not** submit on Gadi as-is.
Use the PBS Pro templates in `jobs/gadi/*.pbs` instead.

Notable bug in old `run_metaphlan.pbs`: `$idlist` unset; R1 FASTQ was passed twice.
Superseded by `jobs/gadi/03_metaphlan_array.pbs`.
