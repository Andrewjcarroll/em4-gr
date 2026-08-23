#!/usr/bin/env bash
# EM4 scaling driver (methods paper): compact FD vs explicit FD strong scaling.
#
# The accuracy story (compact reaches a given error on a coarser mesh) is made
# elsewhere. THIS script makes the performance payoff: at matched accuracy the
# compact scheme uses a narrower stencil, so ghost-exchange communication does
# not grow the way a wider explicit stencil's does. We run EM4's
# `solverScalingTest` (a fixed 5-step micro-run that dumps per-phase min/mean/max
# wall times) across MPI rank counts for several derivative schemes, then plot
# evolve time and the communication cost (the gap between the "with comm" and
# "compute only" unzip/zip timers).
#
# Build is module-driven so the same script works on a laptop and on HPC.
# Requires the EM4 `main` branch (the string-based DendroDerivatives selection
# is commented out on cuda-build), linking dendrolib_dfvk_copy.
#
# Run `./scripts/run_em4_scaling.sh --help` for usage.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

usage() {
  cat <<EOF
EM4 scaling driver -- builds EM4 then sweeps derivative schemes x MPI ranks.

USAGE
  ./scripts/run_em4_scaling.sh [--help]

  Options are environment variables (defaults in brackets):
    SCHEMES   space list of label:first:second:eleorder tuples
              [E6:E6:E6:6 JTT6:JTT6:JTT6:6 E8:E8:E8:8]
              (E6 vs JTT6 at ele_order 6 = same ghost width; E8 at order 8 =
               wider stencil/more comm but the accuracy JTT6 aims to match)
    RANKS     MPI rank counts to sweep            [1 2 4 8]
    MODE      strong | weak                       [strong]
    MAXDEPTH  octree max depth (strong scaling)   [8]
    WEAK_DEPTHS  per-rank maxdepth list for MODE=weak, same length as RANKS
    BASE_PAR  base param TOML                     [em4_simplified.param.toml]
    LAUNCH    MPI launcher, {NP} is substituted   [mpirun -np {NP}]
              HPC example: LAUNCH="srun -n {NP} --cpu-bind=cores"
    TS_MODE   timestepper mode arg to the binary  [1]
    BUILD_DIR build dir                           [build]
    FRESH     1 = wipe BUILD_DIR and reconfigure  [0]
    SKIP_BUILD 1 = use existing binaries          [0]
    JOBS      build parallelism                   [nproc]
    BUILD_TYPE CMake build type                   [Release]
    CMAKE_EXTRA extra cmake configure args        []
    MODULES   'module load' these first (compiler, MKL, MPI)
    OUTDIR    results dir          [scripts/em4_scaling_results]
    PYTHON    python interpreter                  [python3]

OUTPUT
  <OUTDIR>/<scheme>_np<N>.txt   tagged solverCtx_WS dumps (one per run)
  scripts/em4_scaling.csv       tidy parsed results
  scripts/em4_scaling_plots/    strong_scaling, comm_cost, comm_fraction

EXAMPLES
  ./scripts/run_em4_scaling.sh                          # local quick sweep
  RANKS="1 2 4 8 16 32" MAXDEPTH=10 ./scripts/run_em4_scaling.sh
  MODULES="gcc/15 intel-oneapi-mkl/2025 openmpi/5" \\
    LAUNCH="srun -n {NP} --cpu-bind=cores" RANKS="64 128 256 512" \\
    ./scripts/run_em4_scaling.sh
EOF
}
[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && { usage; exit 0; }

SCHEMES=${SCHEMES:-"E6:E6:E6:6 JTT6:JTT6:JTT6:6 E8:E8:E8:8"}
RANKS=${RANKS:-"1 2 4 8"}
MODE=${MODE:-strong}
MAXDEPTH=${MAXDEPTH:-8}
WEAK_DEPTHS=${WEAK_DEPTHS:-}
BASE_PAR=${BASE_PAR:-"$REPO/em4_simplified.param.toml"}
LAUNCH=${LAUNCH:-"mpirun -np {NP}"}
TS_MODE=${TS_MODE:-1}
BUILD_DIR=${BUILD_DIR:-"$REPO/build"}
FRESH=${FRESH:-0}
SKIP_BUILD=${SKIP_BUILD:-0}
JOBS=${JOBS:-$(nproc)}
BUILD_TYPE=${BUILD_TYPE:-Release}
CMAKE_EXTRA=${CMAKE_EXTRA:-}
MODULES=${MODULES:-}
OUTDIR=${OUTDIR:-"$HERE/em4_scaling_results"}
PARDIR="$HERE/em4_scaling_pars"
PYTHON=${PYTHON:-python3}
BIN_NAME=solverScalingTest

# --- modules (toolchain + runtime MKL/MPI) ---------------------------------
if [[ -n "$MODULES" ]] && command -v module >/dev/null 2>&1; then
  echo "## module load $MODULES"
  # shellcheck disable=SC2086
  module load $MODULES || { echo "ERROR: module load failed" >&2; exit 1; }
fi

# --- guard: must be on main (deriv selection is live there) ----------------
BR="$(git -C "$REPO" branch --show-current 2>/dev/null || echo '?')"
if [[ "$BR" != "main" ]]; then
  echo "WARNING: EM4 is on branch '$BR', not 'main'. The string-based" >&2
  echo "         derivative selection (SOLVER_DERIVTYPE_FIRST) is only wired" >&2
  echo "         up on main; compact schemes may be ignored otherwise." >&2
fi

BIN="$BUILD_DIR/solver/$BIN_NAME"

# --- build em4Solver + solverScalingTest -----------------------------------
if [[ "$SKIP_BUILD" == "1" ]]; then
  [[ -x "$BIN" ]] || { echo "ERROR: SKIP_BUILD=1 but $BIN missing" >&2; exit 1; }
else
  command -v cmake >/dev/null || { echo "ERROR: cmake not found" >&2; exit 1; }
  [[ "$FRESH" == "1" ]] && { echo "## FRESH=1 -- wiping $BUILD_DIR"; rm -rf "$BUILD_DIR"; }
  if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    echo "## configuring $BUILD_DIR (SOLVER_PROFILE_SCALING_RUN=ON)"
    # SOLVER_PROFILE_SCALING_RUN defines __PROFILE_ETS__/__PROFILE_CTX__ so
    # solverScalingTest actually writes its per-phase timing table (without it
    # only a banner is emitted) AND disables IO output. CMAKE_POLICY_VERSION_
    # MINIMUM keeps CMake 4.x happy with any transitively-old dependency.
    # shellcheck disable=SC2086
    cmake -S "$REPO" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
          -DSOLVER_PROFILE_SCALING_RUN=ON \
          -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
          -DDENDRO_dendrolib_DIR="$REPO/../dendrolib_dfvk_copy" $CMAKE_EXTRA \
          >"$BUILD_DIR.cfg.log" 2>&1 \
      || { echo "ERROR: configure failed -- see $BUILD_DIR.cfg.log" >&2
           tail -15 "$BUILD_DIR.cfg.log" >&2; exit 1; }
  fi
  for t in em4Solver solverScalingTest; do
    echo "## building $t (-j$JOBS) -> $BUILD_DIR.build.log"
    cmake --build "$BUILD_DIR" --target "$t" -j"$JOBS" \
          >>"$BUILD_DIR.build.log" 2>&1 \
      || { echo "ERROR: build of $t failed -- see $BUILD_DIR.build.log" >&2
           tail -15 "$BUILD_DIR.build.log" >&2; exit 1; }
  done
fi
[[ -x "$BIN" ]] || { echo "ERROR: $BIN not found after build" >&2; exit 1; }
echo "## binary: $BIN"

# --- param generation -------------------------------------------------------
[[ -f "$BASE_PAR" ]] || { echo "ERROR: base par $BASE_PAR missing" >&2; exit 1; }
mkdir -p "$PARDIR" "$OUTDIR"

# set_key FILE KEY VALUE -- replace the RHS of a TOML assignment, matching the
# key whether written bare (SOLVER_DERIVTYPE_FIRST) or prefixed/quoted
# ("dsolve::SOLVER_ELE_ORDER"). VALUE must include quotes if it is a string.
set_key() {
  local f="$1" key="$2" val="$3"
  sed -i -E "s|^([[:space:]]*\"?(dsolve::)?${key}\"?[[:space:]]*=[[:space:]]*).*|\1${val}|" "$f"
}

gen_par() {  # gen_par LABEL FIRST SECOND ELEORDER DEPTH -> path
  local label="$1" first="$2" second="$3" eleord="$4" depth="$5"
  local f="$PARDIR/${label}_d${depth}.param.toml"
  cp "$BASE_PAR" "$f"
  set_key "$f" SOLVER_DERIVTYPE_FIRST  "\"$first\""
  set_key "$f" SOLVER_DERIVTYPE_SECOND "\"$second\""
  set_key "$f" SOLVER_ELE_ORDER        "$eleord"
  set_key "$f" SOLVER_MAXDEPTH          "$depth"
  set_key "$f" SOLVER_PROFILE_FILE_PREFIX "\"em4_${label}_d${depth}\""
  echo "$f"
}

# --- sweep ------------------------------------------------------------------
read -ra _ranks <<< "$RANKS"
read -ra _weak  <<< "$WEAK_DEPTHS"

run_one() {  # run_one LABEL PARFILE NP
  local label="$1" par="$2" np="$3"
  local tagged="$OUTDIR/${label}_np${np}.txt"
  local raw="$OUTDIR/solverCtx_WS_${np}.txt"
  rm -f "$raw" "$tagged"
  local launch="${LAUNCH//\{NP\}/$np}"
  echo "## run: $label  np=$np  ($launch)"
  # solverScalingTest writes solverCtx_WS_<np>.txt to its CWD
  ( cd "$OUTDIR" && eval "$launch \"$BIN\" \"$par\" $TS_MODE" ) \
      >"$OUTDIR/${label}_np${np}.log" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "   WARN: run failed (rc=$rc) -- see ${label}_np${np}.log" >&2
    return 1
  fi
  [[ -f "$raw" ]] && mv "$raw" "$tagged" \
      || { echo "   WARN: no $raw produced" >&2; return 1; }
}

for tuple in $SCHEMES; do
  IFS=':' read -r label first second eleord <<< "$tuple"
  for i in "${!_ranks[@]}"; do
    np="${_ranks[$i]}"
    if [[ "$MODE" == "weak" ]]; then
      depth="${_weak[$i]:-$MAXDEPTH}"
    else
      depth="$MAXDEPTH"
    fi
    par="$(gen_par "$label" "$first" "$second" "$eleord" "$depth")"
    run_one "$label" "$par" "$np" || true
  done
done

# --- parse + plot -----------------------------------------------------------
echo "## em4_scaling_plot.py"
"$PYTHON" "$HERE/em4_scaling_plot.py" "$OUTDIR" \
    --out-dir "$HERE/em4_scaling_plots" --csv "$HERE/em4_scaling.csv" \
    || echo "WARN: em4_scaling_plot.py failed" >&2

echo "## done. tagged results in $OUTDIR ; plots in scripts/em4_scaling_plots/"
