#!/usr/bin/env bash
# EM4 wide-prolongation / buffer-layer A/B driver (deep dipole).
#
# Generates a parfile from scripts/em4_wpx_pars/deep_dipole.template.toml and
# runs ONE configuration, logging to OUTDIR/<label>.log (+ the solver's
# <label>_ANALYTICAL_DIFF.csv in OUTDIR). Configurations differ by the BINARY
# (narrow vs wide-graded build) and SOLVER_REFINE_BUFFER_LAYERS.
#
#   BIN=build_wpx_narrow/solver/em4Solver LABEL=narrow   BUFFER=0 ./scripts/run_em4_wpx.sh
#   BIN=build_wpx_wide/solver/em4Solver   LABEL=wide     BUFFER=0 ./scripts/run_em4_wpx.sh
#   BIN=build_wpx_narrow/solver/em4Solver LABEL=narrow_b1 BUFFER=1 ./scripts/run_em4_wpx.sh
#
# Env knobs (defaults in brackets):
#   BIN, LABEL, BUFFER[0], NP[8], LAUNCH["mpirun -np {NP}"], TS_MODE[1],
#   TSOUT[1] (terminal output freq), OUTDIR[scripts/em4_wpx_results],
#   TEND / MAXDEPTH / WTOL / LAMBDA / AMP / REMESH_FREQ (override the
#   template's RK_TIME_END / SOLVER_MAXDEPTH / SOLVER_WAVELET_TOL /
#   EM4_ID_LAMBDA1 / EM4_ID_AMP1 / SOLVER_REMESH_TEST_FREQ when set).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
BIN=${BIN:?set BIN}
LABEL=${LABEL:?set LABEL}
BUFFER=${BUFFER:-0}
NP=${NP:-8}
LAUNCH=${LAUNCH:-"mpirun -np {NP}"}
TS_MODE=${TS_MODE:-1}
TSOUT=${TSOUT:-1}
OUTDIR=${OUTDIR:-"$HERE/em4_wpx_results"}
TEMPLATE="$HERE/em4_wpx_pars/deep_dipole.template.toml"

[[ -x "$BIN" ]] || BIN="$REPO/$BIN"
[[ -x "$BIN" ]] || { echo "ERROR: binary $BIN not found" >&2; exit 1; }
BIN="$(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")"
mkdir -p "$OUTDIR/vtu" "$OUTDIR/cp"
PAR="$OUTDIR/$LABEL.param.toml"
SEDX=(-e "s|__PREFIX__|$LABEL|g" -e "s|__BUFFER__|$BUFFER|g" -e "s|__TSOUT__|$TSOUT|g")
setkey() { SEDX+=(-e "s|^\"dsolve::$1\" = .*|\"dsolve::$1\" = $2|"); }
[[ -n "${TEND:-}" ]]        && setkey SOLVER_RK_TIME_END "$TEND"
[[ -n "${MAXDEPTH:-}" ]]    && setkey SOLVER_MAXDEPTH "$MAXDEPTH"
[[ -n "${WTOL:-}" ]]        && setkey SOLVER_WAVELET_TOL "$WTOL"
[[ -n "${LAMBDA:-}" ]]      && setkey EM4_ID_LAMBDA1 "$LAMBDA"
[[ -n "${AMP:-}" ]]         && setkey EM4_ID_AMP1 "$AMP"
[[ -n "${REMESH_FREQ:-}" ]] && setkey SOLVER_REMESH_TEST_FREQ "$REMESH_FREQ"
[[ -n "${CFL:-}" ]]         && setkey SOLVER_CFL_FACTOR "$CFL"
[[ -n "${KOSIG:-}" ]]       && setkey KO_DISS_SIGMA "$KOSIG"
# derivative scheme strings (top-of-file keys, e.g. DERIV1=JTT6 DERIV2=JTT6)
[[ -n "${DERIV1:-}" ]] && SEDX+=(-e "s|^SOLVER_DERIVTYPE_FIRST = .*|SOLVER_DERIVTYPE_FIRST = \"$DERIV1\"|")
[[ -n "${DERIV2:-}" ]] && SEDX+=(-e "s|^SOLVER_DERIVTYPE_SECOND = .*|SOLVER_DERIVTYPE_SECOND = \"$DERIV2\"|")
sed "${SEDX[@]}" "$TEMPLATE" > "$PAR"
rm -f "$OUTDIR/${LABEL}_ANALYTICAL_DIFF.csv"
launch=${LAUNCH//\{NP\}/$NP}
echo "## $LABEL: $launch $BIN $PAR $TS_MODE  (OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset})"
start=$(date +%s)
( cd "$OUTDIR" && eval "$launch \"$BIN\" \"$PAR\" $TS_MODE" ) > "$OUTDIR/$LABEL.log" 2>&1
rc=$?
end=$(date +%s)
echo "## $LABEL: rc=$rc wall=$((end-start))s  log=$OUTDIR/$LABEL.log"
exit $rc
