#!/bin/bash
#SBATCH --job-name=em4-wpx-ab
#SBATCH --partition=physics2
#SBATCH --qos=physics
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#
# EM4 deep-dipole A/B on one Zen3 node, pure MPI, OMP_NUM_THREADS=1, the four
# configurations run back-to-back on the same node:
#   A narrow            (build_wpx_narrow, buffer 0)
#   B wide-graded       (build_wpx_wide,   buffer 0)
#   C narrow + 1 buffer (build_wpx_narrow, buffer 1)
#   D wide   + 1 buffer (build_wpx_wide,   buffer 1)
# Knobs: NP (ranks, default 32), TEND (default 8), TSOUT (terminal-output
# freq, default 2), WTOL/MAXDEPTH/LAMBDA/AMP passthrough to run_em4_wpx.sh.
set -uo pipefail
source ~/gr_env.sh >/dev/null 2>&1
EM4=${EM4:-$HOME/research/em4-gr-wpx}
export NP=${NP:-32}
export TEND=${TEND:-8}
export TSOUT=${TSOUT:-2}
export OMP_NUM_THREADS=1
export OUTDIR=${OUTDIR:-$EM4/scripts/em4_wpx_results/${SLURM_JOB_ID}_np${NP}}
export LAUNCH="mpirun --np {NP} --map-by ppr:${NP}:node:pe=1 --bind-to core"
export TS_MODE=1
mkdir -p "$OUTDIR"
echo "job $SLURM_JOB_ID node $(hostname) NP=$NP TEND=$TEND TSOUT=$TSOUT OUTDIR=$OUTDIR"
lscpu | grep -E 'Model name|^CPU\(s\)|NUMA node\(s\)'
cd "$EM4"
run() { # label bin buffer
  BIN="$2" LABEL="$1" BUFFER="$3" ./scripts/run_em4_wpx.sh
  echo "   $1: $(grep 'ETS time' "$OUTDIR/$1.log" | tail -1)  steps=$(grep -c 'Current Step:' "$OUTDIR/$1.log")  remeshes=$(grep -c 'Remesh triggered) ->  old mesh :' "$OUTDIR/$1.log")"
}
run narrow    build_wpx_narrow/solver/em4Solver 0
run wide      build_wpx_wide/solver/em4Solver   0
run narrow_b1 build_wpx_narrow/solver/em4Solver 1
run wide_b1   build_wpx_wide/solver/em4Solver   1
echo "ALL_DONE $OUTDIR"
