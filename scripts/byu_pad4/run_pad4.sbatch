#!/bin/bash
#SBATCH --job-name=em4-pad4
#SBATCH --partition=physics2
#SBATCH --qos=physics
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
#
# Paired A/B for the wide-padding prototype (DENDRO_WIDE_PADDING) on the
# frozen-mesh EM4 dipole sweep: same source tree, same parfiles, one job per
# build arm.  ARM=stock builds with the flag OFF (the original code path),
# ARM=wide with it ON.  Submit both, second with --dependency=afterany:<first>
# so only one runs at a time (cluster-hygiene rule).
#
#   ARM=stock sbatch scripts/byu_pad4/run_pad4.sbatch
#   ARM=wide  sbatch --dependency=afterany:<jobid> scripts/byu_pad4/run_pad4.sbatch
#
# Needs: em4-gr and dendrolib checkouts on branch pad4-prototype (EM4, DL).
# Builds in-job (znver3, profile counters ON so the ctx timers and *_im.prof
# are written).  Runs E6 (control: must be identical across arms) and
# BYU_A6_1ST_R060_OP5 at k = KLIST uniform refinements of the frozen mesh,
# with the [ifc] interface bins and the [census] face census printed.
set -uo pipefail
source ~/gr_env.sh >/dev/null 2>&1
ARM=${ARM:-stock}
EM4=${EM4:-$HOME/research/em4-gr-pad4}
DL=${DL:-$HOME/research/dendrolib_pad4}
NP=${NP:-32}
KLIST=${KLIST:-"2 3"}
SCHEMES=${SCHEMES:-"E6:E6 A6:BYU_A6_1ST_R060_OP5"}
export OMP_NUM_THREADS=1
BASE_PAR="$EM4/scripts/em4_convergence_pars/base.param.toml"
OUTDIR="$EM4/scripts/em4_pad4_results/${SLURM_JOB_ID}_${ARM}_np${NP}"
BUILD="$EM4/build_pad4_${ARM}"
mkdir -p "$OUTDIR/vtu" "$OUTDIR/cp"

echo "job $SLURM_JOB_ID arm $ARM node $(hostname) NP=$NP KLIST=$KLIST"
lscpu | grep -E 'Model name|^CPU\(s\)|NUMA node\(s\)'
git -C "$EM4" log --oneline -1; git -C "$DL" log --oneline -1
[[ $ARM == wide ]] && FLAG=ON || FLAG=OFF

cmake -S "$EM4" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DDENDRO_dendrolib_DIR="$DL" -DDENDRO_WIDE_PADDING=$FLAG \
  -DDENDRO_CPU_ARCH=znver3 -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_Fortran_COMPILER=gfortran \
  -DEM4_COMPUTE_ANALYTICAL=ON -DSOLVER_COMPUTE_CONSTRAINTS=ON -DEM4_ENABLE_COMPACT_DERIVS=ON \
  -DENABLE_DENDRO_PROFILE_COUNTERS=ON -DSOLVER_PROFILE_HUMAN_READABLE=ON \
  -DDENDRO_HYBRID_OMP=OFF -DDENDRO_UNZIP_OMP=OFF -DDENDRO_UNZIP_SPEEDUP=OFF \
  > "$OUTDIR/cfg.log" 2>&1 || { echo "configure failed"; tail -30 "$OUTDIR/cfg.log"; exit 1; }
grep -E '^DENDRO_WIDE_PADDING:|^DENDRO_CPU_ARCH:|^ENABLE_DENDRO_PROFILE_COUNTERS:|^DENDRO_UNZIP_SPEEDUP:' "$BUILD/CMakeCache.txt"
grep -q "Manually-specified variables were not used" "$OUTDIR/cfg.log" && { echo "ERROR: a -D flag was ignored"; grep -A5 "Manually-specified" "$OUTDIR/cfg.log"; exit 1; }
cmake --build "$BUILD" -j 32 --target em4Solver > "$OUTDIR/build.log" 2>&1 || { echo "build failed"; grep -n ' error' "$OUTDIR/build.log" | head; exit 1; }
BIN="$BUILD/solver/em4Solver"

BASE_MAXDEPTH=$(grep '"dsolve::SOLVER_MAXDEPTH"' "$BASE_PAR" | grep -oE '[0-9]+')
BASE_CFL=$(grep '"dsolve::SOLVER_CFL_FACTOR"' "$BASE_PAR" | grep -oE '[0-9.]+$')
cd "$OUTDIR"
for k in $KLIST; do
  for sch in $SCHEMES; do
    label="${sch%%:*}"; deriv="${sch##*:}"
    md=$((BASE_MAXDEPTH + k)); freq=$((4 ** k))
    cfl=$(python3 -c "print($BASE_CFL / 2**$k)")
    par="$OUTDIR/${label}_k${k}.param.toml"
    sed -e "s|^SOLVER_DERIVTYPE_FIRST = .*|SOLVER_DERIVTYPE_FIRST = \"$deriv\"|" \
        -e "s|^\"dsolve::SOLVER_MAXDEPTH\" = .*|\"dsolve::SOLVER_MAXDEPTH\" = $md|" \
        -e "s|^\"dsolve::SOLVER_CFL_FACTOR\" = .*|\"dsolve::SOLVER_CFL_FACTOR\" = $cfl|" \
        -e "s|^\"dsolve::SOLVER_TIME_STEP_OUTPUT_FREQ\" = .*|\"dsolve::SOLVER_TIME_STEP_OUTPUT_FREQ\" = $freq|" \
        -e "s|^\"dsolve::SOLVER_INIT_GRID_UNIFORM_REFINE\" = .*|\"dsolve::SOLVER_INIT_GRID_UNIFORM_REFINE\" = $k|" \
        -e "s|^\"dsolve::SOLVER_PROFILE_FILE_PREFIX\" = .*|\"dsolve::SOLVER_PROFILE_FILE_PREFIX\" = \"${ARM}_${label}_k${k}\"|" \
        "$BASE_PAR" > "$par"
    log="$OUTDIR/${label}_k${k}.log"
    echo "== $ARM $label k=$k (maxdepth $md, cfl $cfl, ifc freq $freq)  $(date)"
    mpirun --np $NP --map-by ppr:${NP}:node:pe=1 --bind-to core "$BIN" "$par" 1 > "$log" 2>&1
    echo "   rc=$?  $(grep 'DERIVS USE' "$log" | head -1 | sed 's/.*DERIVS USE: //' | cut -c1-60)"
    echo "   $(grep 'ETS time' "$log")  steps=$(grep -c 'Current Step:' "$log")"
    grep -A20 "$(grep '^\[ifc\] step' "$log" | tail -1 | cut -d' ' -f1-3)" "$log" | grep -E '\|dE\|  all|divE  all' | cut -c1-200
    grep '^\[census\] all' "$log" | head -1
  done
done
echo "ALL_DONE $OUTDIR"
