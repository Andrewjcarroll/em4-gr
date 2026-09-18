#!/bin/bash
#SBATCH -J x4_pad4_build
#SBATCH -A dwn8
#SBATCH -p m12
#SBATCH --qos=normal
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH -t 00:40:00
#SBATCH --mem=0
#SBATCH -o /home/fslcollab318/research/cfd_bench/x4pad4_build_%j.out
#
# Builds solverScalingTest in both pad4 build dirs.  The suite
# (run_pad4_suite.sbatch) only builds em4Solver, and the ladder needs the
# five-step micro-run binary.  Both dirs are already configured on the login
# node -- this only adds a target, so FETCHCONTENT stays disconnected.
set -uo pipefail
source ~/gr_env.sh >/dev/null 2>&1
R=$HOME/research
EM4=${EM4:-$R/em4-gr-pad4}
DL=${DL:-$R/dendrolib_pad4}
git -C "$EM4" log -1 --oneline; git -C "$DL" log -1 --oneline
for arm in stock wide; do
  B="$EM4/build_pad4_$arm"
  echo "=== $arm  $(grep -E '^DENDRO_WIDE_PADDING:' $B/CMakeCache.txt)"
  cmake --build "$B" -j 32 --target solverScalingTest > "$B.scaling_build.log" 2>&1 \
    || { echo "BUILD FAILED ($arm)"; grep -n ' error' "$B.scaling_build.log" | head; exit 1; }
  ls -l "$B/solver/solverScalingTest"
done
echo BUILD_OK
