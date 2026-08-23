#!/usr/bin/env bash
# Build the two EM4 wpx binaries on the BYU LOGIN node (znver3 target).
#   narrow: DENDRO_WIDE_PROLONGATION=OFF            -> build_wpx_narrow
#   wide:   DENDRO_WIDE_PROLONGATION=ON + COARSER=ON -> build_wpx_wide
# Pure MPI (DENDRO_HYBRID_OMP=OFF), same flags as the local smoke builds.
set -uo pipefail
source ~/gr_env.sh >/dev/null 2>&1
EM4=${EM4:-$HOME/research/em4-gr-wpx}
DL=${DL:-$HOME/research/dendrolib_wpx}
JOBS=${JOBS:-16}
cd "$EM4"
COMMON="-DCMAKE_BUILD_TYPE=Release -DDENDRO_CPU_ARCH=znver3 -DDENDRO_dendrolib_DIR=$DL \
  -DEM4_COMPUTE_ANALYTICAL=ON -DSOLVER_COMPUTE_CONSTRAINTS=ON \
  -DDENDRO_HYBRID_OMP=OFF -DDENDRO_UNZIP_OMP=OFF \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_Fortran_COMPILER=gfortran"
for cfg in narrow wide; do
  if [[ $cfg == narrow ]]; then W="-DDENDRO_WIDE_PROLONGATION=OFF -DDENDRO_WIDE_PROLONGATION_COARSER=OFF"
  else W="-DDENDRO_WIDE_PROLONGATION=ON -DDENDRO_WIDE_PROLONGATION_COARSER=ON"; fi
  echo "=== configure $cfg"
  cmake -S . -B build_wpx_$cfg $COMMON $W > build_wpx_$cfg.cfg.log 2>&1 || { echo "CFG FAIL $cfg"; tail -30 build_wpx_$cfg.cfg.log; exit 1; }
  grep -E '^DENDRO_WIDE_PROLONGATION(_COARSER)?:|^DENDRO_CPU_ARCH:|^DENDRO_HYBRID_OMP:' build_wpx_$cfg/CMakeCache.txt
done
echo "=== build (both, -j$JOBS)"
( cmake --build build_wpx_narrow -j$JOBS --target em4Solver > build_wpx_narrow.build.log 2>&1; echo "narrow rc=$?" ) &
( cmake --build build_wpx_wide   -j$JOBS --target em4Solver > build_wpx_wide.build.log 2>&1;   echo "wide rc=$?" ) &
wait
ls -la build_wpx_narrow/solver/em4Solver build_wpx_wide/solver/em4Solver
grep -n ' error' build_wpx_narrow.build.log build_wpx_wide.build.log | head
echo BUILD_DONE
