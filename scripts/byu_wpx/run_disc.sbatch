#!/bin/bash
#SBATCH --job-name=em4-wpx-disc
#SBATCH --partition=physics2
#SBATCH --qos=physics
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#
# Discriminator runs for the EM4 error floor (narrow build, same deep-dipole
# parfile as run_wpx.sbatch): CFL 0.125 (RK4 error /16 if time-limited) and
# KO sigma 0.1 (dissipation-limited?).
set -uo pipefail
source ~/gr_env.sh >/dev/null 2>&1
EM4=${EM4:-$HOME/research/em4-gr-wpx}
export NP=${NP:-32}
export TEND=${TEND:-8}
export TSOUT=${TSOUT:-2}
export OMP_NUM_THREADS=1
export OUTDIR=${OUTDIR:-$EM4/scripts/em4_wpx_results/${SLURM_JOB_ID}_disc_np${NP}}
export LAUNCH="mpirun --np {NP} --map-by ppr:${NP}:node:pe=1 --bind-to core"
export TS_MODE=1
mkdir -p "$OUTDIR"
echo "job $SLURM_JOB_ID node $(hostname) NP=$NP TEND=$TEND OUTDIR=$OUTDIR"
cd "$EM4"
B=build_wpx_narrow/solver/em4Solver
CFL=0.125 TSOUT=4 BIN=$B LABEL=narrow_cfl0125 BUFFER=0 ./scripts/run_em4_wpx.sh
echo "   narrow_cfl0125: $(grep 'ETS time' $OUTDIR/narrow_cfl0125.log | tail -1) steps=$(grep -c 'Current Step:' $OUTDIR/narrow_cfl0125.log)"
KOSIG=0.1 BIN=$B LABEL=narrow_ko01 BUFFER=0 ./scripts/run_em4_wpx.sh
echo "   narrow_ko01: $(grep 'ETS time' $OUTDIR/narrow_ko01.log | tail -1) steps=$(grep -c 'Current Step:' $OUTDIR/narrow_ko01.log)"
echo "ALL_DONE $OUTDIR"
