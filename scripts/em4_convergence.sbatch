#!/bin/bash
#SBATCH --job-name=em4-mesh-convergence
#SBATCH --partition=physics2
#SBATCH --qos=physics
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
#
# BYU reproduction of the fixed-structure mesh-convergence sweep (paper
# X2 record, 2026-08-31): E6 and JTT6, k = 0..2 uniform refinements of a
# frozen five-level adaptive mesh, errors vs the exact superposed-dipole
# solution. No timing is measured; the numbers are discretization errors and
# should reproduce the archived record to all printed digits on any machine.
#
# Needs: this repo on branch em4-convergence-sweep, and a dendrolib checkout
# of `experimental` at or past the KO-sign/left-right fixes (DL below).
# The driver builds its own binary (build_conv) inside the job.
#
#   sbatch scripts/em4_convergence.sbatch
#
# NOTE: intentionally NOT `--mem=0` -- on m12 that quadruples the billed cost
# (see the cluster-hygiene notes); the default per-node memory is plenty here.
set -uo pipefail
source ~/gr_env.sh >/dev/null 2>&1
EM4=${EM4:-$HOME/research/em4-gr}
DL=${DL:-$HOME/research/dendrolib}
export NP=${NP:-32}
export OMP_NUM_THREADS=1
export DENDROLIB="$DL"
export CMAKE_EXTRA="-DDENDRO_CPU_ARCH=znver3 -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_Fortran_COMPILER=gfortran"
export LAUNCH="mpirun --np {NP} --map-by ppr:${NP}:node:pe=1 --bind-to core"
export OUTDIR="$EM4/scripts/em4_convergence_results/${SLURM_JOB_ID}"
export FRESH=${FRESH:-1}

echo "job $SLURM_JOB_ID node $(hostname) NP=$NP EM4=$EM4 DL=$DL"
lscpu | grep -E 'Model name|^CPU\(s\)|NUMA node\(s\)'
cd "$EM4"
git log --oneline -1
git -C "$DL" log --oneline -1

SCHEMES="E6:E6 JTT6:JTT6" KMAX=${KMAX:-2} ./scripts/run_em4_convergence.sh
