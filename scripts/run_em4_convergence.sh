#!/usr/bin/env bash
# EM4 fixed-structure mesh convergence sweep (methods paper, X2 closure).
#
# Evolves superposed dipole initial data (SOLVER_ID_TYPE=2, exact by
# linearity) on a FROZEN adaptive octree: the base structure is built once by
# function2Octree at a pinned wavelet ceiling, then every octant is refined k
# times (SOLVER_INIT_GRID_UNIFORM_REFINE), so the refinement pattern is
# identical across the sweep and h halves each step -- a genuine h-sweep on a
# genuinely multi-level AMR grid. SOLVER_REFINEMENT_MODE=1 keeps the mesh
# frozen through the evolution.
#
# Time alignment: CFL_k = CFL0 * 2^-k, and dt is CFL*dx(lmax), so
# dt_k = dt_0/4^k exactly; with TIME_STEP_OUTPUT_FREQ = 4^k every run prints
# its interface norms at the same physical times, and the parser compares the
# last common one. RK4 error then scales as h^8, below the h^6 target.
#
# Dissipation is OFF (KO_DISS_SIGMA=0): any KO term is a consistent O(h^2r)
# perturbation of the operator under test.
#
# Usage: ./scripts/run_em4_convergence.sh   (options via env, see below)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

SCHEMES="${SCHEMES:-E6:E6 JTT6:JTT6}"   # label:first-deriv pairs
KMAX="${KMAX:-2}"                        # refinement rounds 0..KMAX
NP="${NP:-4}"
BASE_PAR="${BASE_PAR:-$HERE/em4_convergence_pars/base.param.toml}"
BUILD_DIR="${BUILD_DIR:-$REPO/build_conv}"
OUTDIR="${OUTDIR:-$HERE/em4_convergence_results}"
LAUNCH="${LAUNCH:-mpirun -np {NP}}"
JOBS="${JOBS:-$(nproc)}"
SKIP_BUILD="${SKIP_BUILD:-0}"
FRESH="${FRESH:-0}"
DENDROLIB="${DENDROLIB:-$HOME/research/dendrolib_dfvk_copy}"

BASE_MAXDEPTH=$(grep '"dsolve::SOLVER_MAXDEPTH"' "$BASE_PAR" | grep -oE '[0-9]+')
BASE_CFL=$(grep '"dsolve::SOLVER_CFL_FACTOR"' "$BASE_PAR" | grep -oE '[0-9.]+$')

if [[ "$FRESH" == "1" ]]; then rm -rf "$BUILD_DIR"; fi
if [[ "$SKIP_BUILD" != "1" ]]; then
  cmake -S "$REPO" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
    -DEM4_COMPUTE_ANALYTICAL=ON \
    -DDENDRO_dendrolib_DIR="$DENDROLIB" >"$BUILD_DIR.cfg.log" 2>&1 \
    || { echo "configure failed, see $BUILD_DIR.cfg.log"; exit 1; }
  cmake --build "$BUILD_DIR" -j "$JOBS" --target em4Solver \
    >"$BUILD_DIR.build.log" 2>&1 \
    || { echo "build failed, see $BUILD_DIR.build.log"; exit 1; }
fi
BIN=$(find "$BUILD_DIR" -maxdepth 3 -name "em4Solver" -type f | head -1)
[[ -n "$BIN" ]] || { echo "em4Solver binary not found in $BUILD_DIR"; exit 1; }

mkdir -p "$OUTDIR"
cd "$OUTDIR"
mkdir -p vtu cp

for sch in $SCHEMES; do
  label="${sch%%:*}"; deriv="${sch##*:}"
  for k in $(seq 0 "$KMAX"); do
    par="$OUTDIR/${label}_k${k}.param.toml"
    md=$((BASE_MAXDEPTH + k))
    freq=$((4 ** k))
    # dt ends up CFL*dx(refined lmax), i.e. dx already halves per k, so
    # CFL/2^k gives dt_k = dt_0/4^k: aligned output times at freq 4^k, and
    # RK4 error ~ dt^4 ~ h^8, below the h^6 target
    cfl=$(python3 -c "print($BASE_CFL / 2**$k)")
    sed -e "s|^SOLVER_DERIVTYPE_FIRST = .*|SOLVER_DERIVTYPE_FIRST = \"$deriv\"|" \
        -e "s|^\"dsolve::SOLVER_MAXDEPTH\" = .*|\"dsolve::SOLVER_MAXDEPTH\" = $md|" \
        -e "s|^\"dsolve::SOLVER_CFL_FACTOR\" = .*|\"dsolve::SOLVER_CFL_FACTOR\" = $cfl|" \
        -e "s|^\"dsolve::SOLVER_TIME_STEP_OUTPUT_FREQ\" = .*|\"dsolve::SOLVER_TIME_STEP_OUTPUT_FREQ\" = $freq|" \
        -e "s|^\"dsolve::SOLVER_INIT_GRID_UNIFORM_REFINE\" = .*|\"dsolve::SOLVER_INIT_GRID_UNIFORM_REFINE\" = $k|" \
        -e "s|^\"dsolve::SOLVER_PROFILE_FILE_PREFIX\" = .*|\"dsolve::SOLVER_PROFILE_FILE_PREFIX\" = \"em4_conv_${label}_k${k}\"|" \
        "$BASE_PAR" > "$par"
    log="$OUTDIR/${label}_k${k}.log"
    if [[ -s "$log" ]] && grep -q "ETS time (max)" "$log"; then
      echo "== $label k=$k already done, skipping (rm $log to rerun)"
      continue
    fi
    echo "== $label k=$k (maxdepth $md, cfl $cfl, ifc freq $freq)"
    ${LAUNCH//\{NP\}/$NP} "$BIN" "$par" 1 2>&1 | tee "$log" | \
      grep -E "uniform refinement|Total elements|lmin|elems" | tail -4
  done
done

echo
python3 "$HERE/parse_em4_convergence.py" "$OUTDIR" $SCHEMES --kmax "$KMAX"
