#!/usr/bin/env bash
# Submit the R4 ladder: one job per (rank count, repeat), as the archived X4
# sweep was run.  Ten small jobs schedule far better than one wide one under
# the account's billing-minutes cap, and the low rank counts start at once.
# Walltime is 1.5x the archived per-point budget because there are three arms
# instead of two.
set -euo pipefail
CB=/home/fslcollab318/research/cfd_bench
SB=/apps/slurm/latest/bin/sbatch
JOB=${JOB:-$CB/x4_pad4_ladder.sbatch}
DEP=${DEP:-}                       # e.g. DEP=afterok:<build job id>
declare -A NODES=( [64]=1 [128]=1 [256]=2 [512]=4 [1024]=8 )
declare -A TPN=(   [64]=64 [128]=128 [256]=128 [512]=128 [1024]=128 )
declare -A WALL=(  [64]=00:20:00 [128]=00:20:00 [256]=00:25:00 [512]=00:30:00 [1024]=00:40:00 )  # a point takes ~2 min; short requests schedule under the billing cap
for REP in ${REPS:-1 2}; do
  for NP in ${RANKS:-64 128 256 512 1024}; do
    $SB -N "${NODES[$NP]}" --ntasks-per-node="${TPN[$NP]}" -t "${WALL[$NP]}" \
        -J "x4p4_${NP}r${REP}" ${DEP:+--dependency=$DEP} \
        --export=ALL,X4_RANKS=$NP,X4_REP=$REP "$JOB"
  done
done
