#!/bin/bash
#SBATCH -J x4_pad4
#SBATCH -A dwn8
#SBATCH -p m12
#SBATCH --qos=normal
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -o /home/fslcollab318/research/cfd_bench/x4pad4_%j.out
#
# R4 -- the X4 strong-scaling ladder with a wide-padding arm.
#
# WHY. The methods paper's cost group has the ring's per-step cost only on
# single-node EM4 runs (+11-15%) and one BSSN q12 point (+13/+15/+32%). The
# claim the section needs is that the ring buys accuracy WITHOUT buying
# communication: ghost exchange is by element, the fourth ring is filled from
# data the rank already owns, so the comm timers must be flat between the stock
# and wide arms while only the compute timers move. That is a per-rank-count
# statement and needs the ladder.
#
# WHAT CHANGES vs the archived 2026-09-09 ladder. The source line: this runs
# em4-gr-pad4 + dendrolib_pad4 (branch pad4-prototype) rather than
# em4-gr-main + dendrolib_cfd_ladder, and the compact arm is the BYU A6
# closure rather than JTT6. The three arms here are internally comparable,
# which is what the wide/stock ratio needs; the archived ladder stays the
# cross-source reference.
#
# m12 nodes are 128 cores (2 x EPYC 7763, 1 thread/core). --exclusive is
# mandatory on m12 (OverSubscribe=YES:4; a co-tenant once inverted the sign of
# this comparison). Binaries must be built first -- x4_pad4_build.sbatch.
set -uo pipefail
source ~/gr_env.sh >/dev/null 2>&1
# A compact arm that fails to construct dumps one core per rank: 0.5 GB x 1024
# ranks filled the home filesystem on the first attempt (2026-09-18).
ulimit -c 0
R=$HOME/research
EM4=${EM4:-$R/em4-gr-pad4}
CB=$R/cfd_bench
A6=${A6:-BYU_A6_1ST_R060_OP5}
export OMP_NUM_THREADS=1

BIN_stock="$EM4/build_pad4_stock/solver/solverScalingTest"
BIN_wide="$EM4/build_pad4_wide/solver/solverScalingTest"
for b in "$BIN_stock" "$BIN_wide"; do
  [ -x "$b" ] || { echo "NO BINARY at $b -- run x4_pad4_build.sbatch first"; exit 1; }
done

NP=${X4_RANKS:?set X4_RANKS}
REP=${X4_REP:?set X4_REP}

RUN_ROOT=$CB/runs/X4-pad4/byu-m12/$(date +%F)-$SLURM_JOB_ID
mkdir -p "$RUN_ROOT"
cp "$0" "$RUN_ROOT/run.sbatch"

echo "node=$(hostname) cpus=$(nproc) ranks=$NP rep=$REP"
lscpu | grep -E 'Model name|^CPU\(s\)'
echo "em4:       $(git -C $EM4 log -1 --oneline)"
echo "dendrolib: $(git -C ${DL:-$R/dendrolib_pad4} log -1 --oneline)"
grep -E '^DENDRO_WIDE_PADDING:' "$EM4"/build_pad4_{stock,wide}/CMakeCache.txt

# Parameter files: the archived X4 params, with only the derivative type
# changed per arm.  maxdepth 13, wtol 1e-8, CFL 0.25, KO 0.4, ele_order 6,
# five steps -- a fixed micro-run, so mesh construction dominates the walltime
# and the per-phase timers are what we read.
# Second derivatives: the registry has no 2ND variant under the A6 name, and
# the EM4 right-hand side takes no second derivatives at all, so every arm
# keeps E6 there -- as the pad4 suite does.  Setting it to the compact name
# aborts every rank with "Failed to create Derivs object".
mk_par() {  # $1 = arm label, $2 = deriv name
  local par="$RUN_ROOT/params_$1.toml"
  sed -e "s|^SOLVER_DERIVTYPE_FIRST = .*|SOLVER_DERIVTYPE_FIRST = \"$2\"|" \
      -e "s|^SOLVER_DERIVTYPE_SECOND = .*|SOLVER_DERIVTYPE_SECOND = \"E6\"|" \
      -e "s|^\"dsolve::SOLVER_PROFILE_FILE_PREFIX\" = .*|\"dsolve::SOLVER_PROFILE_FILE_PREFIX\" = \"prof_$1\"|" \
      -e "s|^\"dsolve::SOLVER_VTU_FILE_PREFIX\" = .*|\"dsolve::SOLVER_VTU_FILE_PREFIX\" = \"vtu/x4_$1\"|" \
      "$CB/params_E6.toml" > "$par"
  grep -E '^SOLVER_DERIVTYPE_(FIRST|SECOND)' "$par"
}
mk_par E6  E6
mk_par A6  "$A6"
mk_par A6w "$A6"

# Arms interleaved inside the repeat so machine drift is shared, not assigned
# to one arm.  arm label -> (build, parfile).
for ARM in E6 A6 A6w; do
  case $ARM in
    E6)  bin=$BIN_stock ;;
    A6)  bin=$BIN_stock ;;
    A6w) bin=$BIN_wide  ;;
  esac
  OUT="$RUN_ROOT/np${NP}_${ARM}_rep${REP}"
  mkdir -p "$OUT/vtu" && cd "$OUT"
  echo "=== ranks=$NP arm=$ARM rep=$REP  $(date +%T)"
  srun -n $NP --cpu-bind=cores "$bin" "$RUN_ROOT/params_$ARM.toml" 1 \
      > "$OUT/stdout.log" 2> "$OUT/stderr.log"
  rc=$?
  echo "    exit=$rc  $(grep -c . "$OUT/stdout.log" 2>/dev/null) lines  $(ls solverCtx_WS_* 2>/dev/null | head -1)"
  [ $rc -ne 0 ] && tail -5 "$OUT/stderr.log"
done

echo "=== run directory"
du -sh "$RUN_ROOT"; ls "$RUN_ROOT"
echo "ALL_DONE $RUN_ROOT"
