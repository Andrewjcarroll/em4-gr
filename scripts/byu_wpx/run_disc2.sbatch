#!/bin/bash
#SBATCH --job-name=em4-wpx-disc2
#SBATCH --partition=physics2
#SBATCH --qos=physics
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out
# Second discriminator pair (narrow build): wtol 1e-8 (mesh-resolution limited?)
# and JTT6 compact derivatives (stencil-constant limited?).
set -uo pipefail
source ~/gr_env.sh >/dev/null 2>&1
EM4=${EM4:-$HOME/research/em4-gr-wpx}
export NP=${NP:-32}
export TEND=${TEND:-8}
export TSOUT=${TSOUT:-2}
export OMP_NUM_THREADS=1
export OUTDIR=${OUTDIR:-$EM4/scripts/em4_wpx_results/${SLURM_JOB_ID}_disc2_np${NP}}
export LAUNCH="mpirun --np {NP} --map-by ppr:${NP}:node:pe=1 --bind-to core"
export TS_MODE=1
mkdir -p "$OUTDIR"
echo "job $SLURM_JOB_ID node $(hostname) NP=$NP TEND=$TEND OUTDIR=$OUTDIR"
cd "$EM4"
B=build_wpx_narrow/solver/em4Solver
WTOL=1e-08 BIN=$B LABEL=narrow_wtol1e8 BUFFER=0 ./scripts/run_em4_wpx.sh
echo "   narrow_wtol1e8: $(grep 'ETS time' $OUTDIR/narrow_wtol1e8.log | tail -1) steps=$(grep -c 'Current Step:' $OUTDIR/narrow_wtol1e8.log)"
DERIV1=JTT6 DERIV2=JTT6 BIN=$B LABEL=narrow_jtt6 BUFFER=0 ./scripts/run_em4_wpx.sh
echo "   narrow_jtt6: $(grep 'ETS time' $OUTDIR/narrow_jtt6.log | tail -1) steps=$(grep -c 'Current Step:' $OUTDIR/narrow_jtt6.log)"
echo "ALL_DONE $OUTDIR"
