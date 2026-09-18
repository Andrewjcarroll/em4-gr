#!/bin/bash
#SBATCH --job-name=em4-pad4-suite
#SBATCH --partition=physics2
#SBATCH --qos=physics
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=05:00:00
#SBATCH --output=%x-%j.out
#
# Wide-padding comparison suite: builds BOTH arms in-job (stock = flag OFF, the
# original code path; wide = DENDRO_WIDE_PADDING ON), same optimisations in
# both (znver3, DENDRO_UNZIP_SPEEDUP, terminal-derivative kernels, profile
# counters), then runs one of two sets with identical parfiles per scheme:
#
#   MODE=frozen  frozen five-level dipole mesh, k=2 uniform refinements, KO off,
#                CFL 0.06 (the convergence-sweep record): E6 + A6 on stock,
#                E6 + A6 + E8 on wide.
#   MODE=paper   the methods-paper EM4 setup (dipole, dynamic wavelet AMR,
#                KO 0.4, CFL 0.25, t_end 8, wtol 1e-6) at MAXDEPTH 8, 9, 10:
#                E6 + A6 on stock, E6 + A6 + E8 on wide.  Depth 10 runs on all
#                128 cores.  Overrides: DEPTHS="8 9", WTOL=1e-7 (wavelet
#                tolerance), CFL=0.125, REMESH_FREQ=20, ARMS="stock E6 E6|stock A6 $A6" (arm label deriv,
#                pipe-separated; $A6 is the compact operator name), AMRFAC=0.004
#                (wavelet coarsen factor; 1/128 is the child/parent wavelet ratio for
#                a degree-6 interpolant, so the default 0.1 flip-flops), DL=<other
#                dendrolib clone> with BUILD_SUFFIX=_x (separate build dirs, configure
#                them on the login node first), TAG=x (result-dir suffix), MPI_X="-x VAR"
#                (forward an env var, e.g. DENDRO_WIDE_PADDING_TRIM_COARSE=1, to the ranks).
#
#   MODE=frozen sbatch scripts/byu_pad4/run_pad4_suite.sbatch
#   MODE=paper  sbatch --dependency=afterany:<id> scripts/byu_pad4/run_pad4_suite.sbatch
#
# Needs a login-node configure of build_pad4_stock and build_pad4_wide first
# (FetchContent needs the network). The E8 arm reads the copy-filled outer
# ring on faces with a finer neighbour (no per-face fallback exists for
# explicit stencils); it is the naive "just use E8" a user would get.
set -uo pipefail
source ~/gr_env.sh >/dev/null 2>&1
MODE=${MODE:-frozen}
EM4=${EM4:-$HOME/research/em4-gr-pad4}
DL=${DL:-$HOME/research/dendrolib_pad4}
A6=${A6:-BYU_A6_1ST_R060_OP5}
export OMP_NUM_THREADS=1
OUTDIR="$EM4/scripts/em4_pad4_results/${SLURM_JOB_ID}_${MODE}${FILTER:+_$FILTER}${KOSIG:+_ko$KOSIG}${WTOL:+_wtol$WTOL}${CFL:+_cfl$CFL}${REMESH_FREQ:+_rm$REMESH_FREQ}${AMRFAC:+_af$AMRFAC}${BUILD_SUFFIX:-}${TAG:+_$TAG}"
mkdir -p "$OUTDIR/vtu" "$OUTDIR/cp"
echo "job $SLURM_JOB_ID mode $MODE node $(hostname)"
lscpu | grep -E 'Model name|^CPU\(s\)'
git -C "$EM4" log --oneline -1; git -C "$DL" log --oneline -1

build_arm() {  # $1 = stock|wide
  local arm=$1 flag=OFF; [[ $arm == wide ]] && flag=ON
  local B="$EM4/build_pad4_${arm}${BUILD_SUFFIX:-}"
  [[ -d "$B/_deps/xsmm-src" ]] || { echo "ERROR: $B not configured on the login node"; exit 1; }
  cmake -S "$EM4" -B "$B" -DCMAKE_BUILD_TYPE=Release -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
    -DDENDRO_dendrolib_DIR="$DL" -DDENDRO_WIDE_PADDING=$flag -DDENDRO_CPU_ARCH=${ARCH:-znver3} \
    -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_Fortran_COMPILER=gfortran \
    -DEM4_COMPUTE_ANALYTICAL=ON -DSOLVER_COMPUTE_CONSTRAINTS=ON -DEM4_ENABLE_COMPACT_DERIVS=ON \
    -DENABLE_DENDRO_PROFILE_COUNTERS=ON -DSOLVER_PROFILE_HUMAN_READABLE=ON \
    -DDENDRO_HYBRID_OMP=OFF -DDENDRO_UNZIP_OMP=OFF -DDENDRO_UNZIP_SPEEDUP=ON \
    > "$OUTDIR/cfg_$arm.log" 2>&1 || { echo "configure $arm failed"; tail -20 "$OUTDIR/cfg_$arm.log"; exit 1; }
  grep -q "Manually-specified variables were not used" "$OUTDIR/cfg_$arm.log" && { echo "ERROR: ignored -D flag ($arm)"; exit 1; }
  grep -q -- '-march=' "$B/solver/CMakeFiles/em4Solver.dir/flags.make" || { echo "ERROR: no -march ($arm)"; exit 1; }
  cmake --build "$B" -j 32 --target em4Solver > "$OUTDIR/build_$arm.log" 2>&1 || { echo "build $arm failed"; grep -n ' error' "$OUTDIR/build_$arm.log" | head; exit 1; }
  grep -E '^DENDRO_WIDE_PADDING:|^DENDRO_UNZIP_SPEEDUP:|^DENDRO_CPU_ARCH:' "$B/CMakeCache.txt" | tr '\n' ' '; echo
}
build_arm stock; build_arm wide
BIN_stock="$EM4/build_pad4_stock${BUILD_SUFFIX:-}/solver/em4Solver"; BIN_wide="$EM4/build_pad4_wide${BUILD_SUFFIX:-}/solver/em4Solver"

report() {  # $1 = log
  local log=$1
  echo "   rc=$2  $(grep 'DERIVS USE' "$log" | head -1 | sed 's/.*DERIVS USE: //' | cut -c1-60)"
  echo "   $(grep 'ETS time' "$log")  steps=$(grep -c 'Current Step:' "$log")  $(grep -o 'elems [0-9]*' "$log" | tail -1)"
  local last; last=$(grep '^\[ifc\] step' "$log" | tail -1 | cut -d' ' -f1-3)
  [[ -n "$last" ]] && grep -A24 "$last" "$log" | grep -E '\|dE\|  all|\|dB\|  all|divE  all|divB  all' | cut -c1-210
  grep '^\[census\] all' "$log" | head -1
}

if [[ $MODE == frozen ]]; then
  NP=${NP:-32}; K=${K:-2}
  BASE_PAR="$EM4/scripts/em4_convergence_pars/base.param.toml"
  BASE_MAXDEPTH=$(grep '"dsolve::SOLVER_MAXDEPTH"' "$BASE_PAR" | grep -oE '[0-9]+')
  BASE_CFL=$(grep '"dsolve::SOLVER_CFL_FACTOR"' "$BASE_PAR" | grep -oE '[0-9.]+$')
  md=$((BASE_MAXDEPTH + K)); freq=$((4 ** K)); cfl=$(python3 -c "print($BASE_CFL / 2**$K)")
  cd "$OUTDIR"
  for cfg in "stock E6 E6" "stock A6 $A6" "wide E6 E6" "wide A6 $A6" "wide E8 E8"; do
    set -- $cfg; arm=$1; label=$2; deriv=$3; bin=BIN_$arm
    par="$OUTDIR/${arm}_${label}_k${K}.param.toml"
    sed -e "s|^SOLVER_DERIVTYPE_FIRST = .*|SOLVER_DERIVTYPE_FIRST = \"$deriv\"|" \
        -e "s|^\"dsolve::SOLVER_MAXDEPTH\" = .*|\"dsolve::SOLVER_MAXDEPTH\" = $md|" \
        -e "s|^\"dsolve::SOLVER_CFL_FACTOR\" = .*|\"dsolve::SOLVER_CFL_FACTOR\" = $cfl|" \
        -e "s|^\"dsolve::SOLVER_TIME_STEP_OUTPUT_FREQ\" = .*|\"dsolve::SOLVER_TIME_STEP_OUTPUT_FREQ\" = $freq|" \
        -e "s|^\"dsolve::SOLVER_INIT_GRID_UNIFORM_REFINE\" = .*|\"dsolve::SOLVER_INIT_GRID_UNIFORM_REFINE\" = $K|" \
        -e "s|^\"dsolve::SOLVER_PROFILE_FILE_PREFIX\" = .*|\"dsolve::SOLVER_PROFILE_FILE_PREFIX\" = \"${arm}_${label}_k${K}\"|" \
        "$BASE_PAR" > "$par"
    log="$OUTDIR/${arm}_${label}_k${K}.log"
    echo "== frozen $arm $label k=$K (maxdepth $md, cfl $cfl, NP $NP)  $(date)"
    mpirun --np $NP --map-by ppr:${NP}:node:pe=1 --bind-to core "${!bin}" "$par" 1 > "$log" 2>&1; rc=$?
    report "$log" $rc
  done
else
  export TEND=${TEND:-8} TSOUT=${TSOUT:-4} OUTDIR TS_MODE=1 KOSIG FILTER WTOL CFL REMESH_FREQ AMRFAC
  ARMS=${ARMS:-"stock E6 E6|stock A6 $A6|wide E6 E6|wide A6 $A6|wide E8 E8"}
  for depth in ${DEPTHS:-8 9 10}; do
    NP=32; [[ $depth -ge 10 ]] && NP=128
    export NP LAUNCH="mpirun ${MPI_X:-} --np {NP} --map-by ppr:${NP}:node:pe=1 --bind-to core" MAXDEPTH=$depth
    IFS="|" read -ra CFGS <<< "$ARMS"
    for cfg in "${CFGS[@]}"; do
      set -- $cfg; arm=$1; label=$2; deriv=$3; bin=BIN_$arm
      echo "== paper $arm $label depth $depth (NP $NP, t_end $TEND)  $(date)"
      BIN="${!bin}" LABEL="${arm}_${label}_d${depth}" BUFFER=0 DERIV1=$deriv DERIV2=E6 "$EM4/scripts/run_em4_wpx.sh" > "$OUTDIR/driver_${arm}_${label}_d${depth}.log" 2>&1; rc=$?
      report "$OUTDIR/${arm}_${label}_d${depth}.log" $rc
    done
  done
fi
echo "ALL_DONE $OUTDIR"
