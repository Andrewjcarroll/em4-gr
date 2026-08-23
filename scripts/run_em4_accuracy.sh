#!/usr/bin/env bash
# EM4 in-simulation accuracy driver (methods paper, demonstration figure).
#
# Evolves the dipole electromagnetic initial data for several derivative schemes
# (explicit E4/E6 vs compact JTT6) and records the L2 error against the analytic
# solution over time -- the paper's Ex_error / Bx_error figures. This shows the
# compact operator works in a LIVE evolution and tracks the explicit schemes.
# The deeper convergence / constraint-violation / long-time-stability study is
# the companion paper's job, not this.
#
# This is a SEPARATE build from run_em4_scaling.sh: it needs
# -DEM4_COMPUTE_ANALYTICAL=ON (writes <prefix>_ANALYTICAL_DIFF.csv) and IO left
# ON, whereas the scaling build uses SOLVER_PROFILE_SCALING_RUN=ON which turns
# IO off. Requires the EM4 `main` branch (string-based scheme selection) linking
# dendrolib_dfvk_copy.
#
# Run `./scripts/run_em4_accuracy.sh --help` for usage.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

usage() {
  cat <<EOF
EM4 in-sim accuracy driver -- evolve dipole per scheme, plot error vs time.

USAGE
  ./scripts/run_em4_accuracy.sh [--help]

  Options are environment variables (defaults in brackets):
    SCHEMES   space list of label:first:second:eleorder tuples
              [E4:E4:E4:6 E6:E6:E6:6 JTT6:JTT6:JTT6:6]
    MAXDEPTH  octree max depth                    [8]
    TEND      evolution end time (let pulse leave grid)  [from base par]
    NP        MPI ranks for the runs              [1]
    VARS      variables to plot (comma)           [U_E0,U_B0]
    METRIC    error column to plot                [DIFF_L2]
    BASE_PAR  base param TOML       [em4_simplified.param.toml]
    LAUNCH    MPI launcher, {NP} substituted      [mpirun -np {NP}]
    TS_MODE   timestepper mode arg                [1]
    BUILD_DIR build dir                           [build_accuracy]
    FRESH     1 = wipe BUILD_DIR + reconfigure    [0]
    SKIP_BUILD 1 = use existing binary            [0]
    JOBS / BUILD_TYPE / CMAKE_EXTRA / MODULES / PYTHON   (as usual)
    OUTDIR    results dir          [scripts/em4_accuracy_results]

OUTPUT
  <OUTDIR>/<scheme>_diff.csv     tagged ANALYTICAL_DIFF dumps
  scripts/em4_accuracy_plots/    <var>_error.png (e.g. U_E0_error, U_B0_error)
  final-time error summary on stdout

EXAMPLES
  ./scripts/run_em4_accuracy.sh                       # E4/E6/JTT6, default res
  MAXDEPTH=9 TEND=30 ./scripts/run_em4_accuracy.sh
  MODULES="gcc mkl openmpi" NP=4 ./scripts/run_em4_accuracy.sh
EOF
}
[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && { usage; exit 0; }

SCHEMES=${SCHEMES:-"E4:E4:E4:6 E6:E6:E6:6 JTT6:JTT6:JTT6:6"}
MAXDEPTH=${MAXDEPTH:-8}
WAVELET_TOL=${WAVELET_TOL:-}
TEND=${TEND:-}
NP=${NP:-1}
VARS=${VARS:-"U_E0,U_B0"}
# volume-aware (mesh-integrated) norm by default -- fair across differing AMR
# meshes; plain per-node DIFF_L2 biases fine vs coarse cells. (DIFF_L2_INT)
METRIC=${METRIC:-DIFF_L2_INT}
BASE_PAR=${BASE_PAR:-"$REPO/em4_simplified.param.toml"}
LAUNCH=${LAUNCH:-"mpirun -np {NP}"}
TS_MODE=${TS_MODE:-1}
BUILD_DIR=${BUILD_DIR:-"$REPO/build_accuracy"}
FRESH=${FRESH:-0}
SKIP_BUILD=${SKIP_BUILD:-0}
JOBS=${JOBS:-$(nproc)}
BUILD_TYPE=${BUILD_TYPE:-Release}
CMAKE_EXTRA=${CMAKE_EXTRA:-}
MODULES=${MODULES:-}
OUTDIR=${OUTDIR:-"$HERE/em4_accuracy_results"}
PARDIR="$HERE/em4_accuracy_pars"
PYTHON=${PYTHON:-python3}
BIN_NAME=em4Solver

if [[ -n "$MODULES" ]] && command -v module >/dev/null 2>&1; then
  echo "## module load $MODULES"
  # shellcheck disable=SC2086
  module load $MODULES || { echo "ERROR: module load failed" >&2; exit 1; }
fi

BR="$(git -C "$REPO" branch --show-current 2>/dev/null || echo '?')"
[[ "$BR" != "main" ]] && echo "WARNING: EM4 on '$BR', not 'main' -- compact scheme selection may be inactive." >&2

BIN="$BUILD_DIR/solver/$BIN_NAME"

# --- build em4Solver with analytic-diff enabled ----------------------------
if [[ "$SKIP_BUILD" == "1" ]]; then
  [[ -x "$BIN" ]] || { echo "ERROR: SKIP_BUILD=1 but $BIN missing" >&2; exit 1; }
else
  command -v cmake >/dev/null || { echo "ERROR: cmake not found" >&2; exit 1; }
  [[ "$FRESH" == "1" ]] && { echo "## FRESH=1 -- wiping $BUILD_DIR"; rm -rf "$BUILD_DIR"; }
  if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    echo "## configuring $BUILD_DIR (EM4_COMPUTE_ANALYTICAL=ON, IO on)"
    # shellcheck disable=SC2086
    cmake -S "$REPO" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
          -DEM4_COMPUTE_ANALYTICAL=ON \
          -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
          -DDENDRO_dendrolib_DIR="$REPO/../dendrolib_dfvk_copy" $CMAKE_EXTRA \
          >"$BUILD_DIR.cfg.log" 2>&1 \
      || { echo "ERROR: configure failed -- see $BUILD_DIR.cfg.log" >&2
           tail -15 "$BUILD_DIR.cfg.log" >&2; exit 1; }
  fi
  echo "## building $BIN_NAME (-j$JOBS) -> $BUILD_DIR.build.log"
  cmake --build "$BUILD_DIR" --target "$BIN_NAME" -j"$JOBS" \
        >"$BUILD_DIR.build.log" 2>&1 \
    || { echo "ERROR: build failed -- see $BUILD_DIR.build.log" >&2
         tail -15 "$BUILD_DIR.build.log" >&2; exit 1; }
fi
[[ -x "$BIN" ]] || { echo "ERROR: $BIN not found after build" >&2; exit 1; }
echo "## binary: $BIN"

# --- param generation -------------------------------------------------------
[[ -f "$BASE_PAR" ]] || { echo "ERROR: base par $BASE_PAR missing" >&2; exit 1; }
mkdir -p "$PARDIR" "$OUTDIR"

set_key() {
  local f="$1" key="$2" val="$3"
  sed -i -E "s|^([[:space:]]*\"?(dsolve::)?${key}\"?[[:space:]]*=[[:space:]]*).*|\1${val}|" "$f"
}

gen_par() {  # gen_par LABEL FIRST SECOND ELEORDER [MATID] -> path
  local label="$1" first="$2" second="$3" eleord="$4" matid="$5"
  local f="$PARDIR/${label}.param.toml"
  cp "$BASE_PAR" "$f"
  set_key "$f" SOLVER_DERIVTYPE_FIRST  "\"$first\""
  set_key "$f" SOLVER_DERIVTYPE_SECOND "\"$second\""
  set_key "$f" SOLVER_ELE_ORDER        "$eleord"
  set_key "$f" SOLVER_MAXDEPTH          "$MAXDEPTH"
  # matrixID: for Boris/Brady schemes it selects the block boundary CLOSURE
  # (1=Dirichlet/lopsided, 2/3=proper ghost-aware closure). Ignored by JTT6.
  # These keys are NOT in the base param, and set_key only REPLACES existing
  # lines, so append them when absent (parsed bare, no dsolve:: prefix).
  if [[ -n "$matid" ]]; then
    local k
    for k in SOLVER_DERIV_FIRST_MATID SOLVER_DERIV_SECOND_MATID; do
      if grep -qE "^[[:space:]]*\"?(dsolve::)?${k}\"?[[:space:]]*=" "$f"; then
        set_key "$f" "$k" "$matid"
      else
        printf '%s = %s\n' "$k" "$matid" >> "$f"
      fi
    done
  fi
  # wavelet tolerance: 1e-5 (default) oscillates the AMR mesh (512<->848) and
  # makes init grid convergence churn; 1e-3 settles cleanly. Settable here.
  [[ -n "$WAVELET_TOL" ]] && set_key "$f" SOLVER_WAVELET_TOL "$WAVELET_TOL"
  set_key "$f" SOLVER_PROFILE_FILE_PREFIX "\"em4_acc_${label}\""
  # disable VTU output: it writes a vtk every IO step into a vtu/ subdir that
  # doesn't exist in the run dir (noisy IO errors + wasted time). The analytic
  # diff CSV is written in terminal_output (TIME_STEP_OUTPUT_FREQ), not here, so
  # accuracy data is unaffected.
  set_key "$f" SOLVER_IO_OUTPUT_FREQ    "1000000"
  if [[ -n "$TEND" ]]; then
    # SOLVER_RK_TIME_END is parsed as a float (as_floating); a bare integer
    # like "30" throws toml::type_error, so force a decimal point.
    local tend="$TEND"
    [[ "$tend" == *.* || "$tend" == *e* || "$tend" == *E* ]] || tend="${tend}.0"
    set_key "$f" SOLVER_RK_TIME_END "$tend"
  fi
  echo "$f"
}

run_one() {  # run_one LABEL PARFILE
  local label="$1" par="$2"
  local diff_src="$OUTDIR/em4_acc_${label}_ANALYTICAL_DIFF.csv"
  local diff_dst="$OUTDIR/${label}_diff.csv"
  rm -f "$diff_src" "$diff_dst"
  local launch="${LAUNCH//\{NP\}/$NP}"
  echo "## run: $label  ($launch, maxdepth=$MAXDEPTH)"
  ( cd "$OUTDIR" && eval "$launch \"$BIN\" \"$par\" $TS_MODE" ) \
      >"$OUTDIR/${label}.log" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "   WARN: run failed (rc=$rc) -- see ${label}.log" >&2; return 1
  fi
  [[ -f "$diff_src" ]] && mv "$diff_src" "$diff_dst" \
      || { echo "   WARN: no ANALYTICAL_DIFF.csv produced (analytic build?)" >&2; return 1; }
}

for tuple in $SCHEMES; do
  IFS=':' read -r label first second eleord matid <<< "$tuple"
  par="$(gen_par "$label" "$first" "$second" "$eleord" "$matid")"
  run_one "$label" "$par" || true
done

# --- parse + plot -----------------------------------------------------------
echo "## em4_accuracy_plot.py"
"$PYTHON" "$HERE/em4_accuracy_plot.py" "$OUTDIR" \
    --vars "$VARS" --metric "$METRIC" \
    --out-dir "$HERE/em4_accuracy_plots" \
    || echo "WARN: em4_accuracy_plot.py failed" >&2

echo "## done. tagged results in $OUTDIR ; plots in scripts/em4_accuracy_plots/"
