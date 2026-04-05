# CMAKE CONFIGURATION FILE FOR ENTIRE PROJECT
#
# This file in particular contains all of the options that apply to all aspects
# of the program, including some Dendro options that are used when including the
# library.
#
# Note that all options and compiler flags set here will propagate to other
# directories such as src/ and dendro/

# NOTE: cmake minimum is now set to 3.1
cmake_minimum_required(VERSION 3.15)

# full project name and the languages used
project(DendroEM4 C CXX Fortran)

if(POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)
endif()

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# OpenMP, MPI, and GSL are required packages for all aspects of the project
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(GSL REQUIRED)

# =============================
# == CPU ARCH OPTIONS ==
# =============================
set(CPU_ARCH "native" CACHE STRING "Target CPU Architecture, see list")

# this list of strings are the valid list of options we have available for compiling
set(CPU_ARCH_STRINGS
    "native"
    "znver1" "znver2" "znver3" "znver4"
    "haswell" "broadwell" "skylake-avx512" "cascadelake" "icelake-server"
    "generic_avx2" "x86-64-v3"
)
set_property(CACHE CPU_ARCH PROPERTY STRINGS ${CPU_ARCH_STRINGS})

set(ARCH_FLAGS "")

if(CPU_ARCH STREQUAL "generic_avx2")
  # this is like a custom alias for generic avx2
  list(APPEND ARCH_FLAGS "-mavx2" "-mfma")
elseif(CPU_ARCH IN_LIST CPU_ARCH_STRINGS)
  # we pass the name directly to "march" if it's in the list of strings
  list(APPEND ARCH_FLAGS "-march=${CPU_ARCH}")
else()
  message(FATAL_ERROR "Unknown CPU_ARCH: ${CPU_ARCH}")
endif()

# by setting compile options, we're forcing things to be set properly
add_compile_options(${ARCH_FLAGS})

message(STATUS "Build Target Architecture: ${CPU_ARCH}")
message(STATUS "Compiler Flags: ${ARCH_FLAGS}")

# set up fsanatize flags to check for memory leaks (Debug only)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set(SAN_FLAGS "-fsanitize=address,undefined" "-fno-sanitize-recover=all")
  foreach(FLAG ${SAN_FLAGS})
    add_compile_options($<$<CONFIG:Debug>:${FLAG}>)
    add_link_options($<$<CONFIG:Debug>:${FLAG}>)
  endforeach()
  message(STATUS "Sanitizers configured for Debug builds (Address + Undefined)")
endif()

# =============================
# == DENDRO REQUIRED OPTIONS ==
# =============================
option(USE_64BIT_INDICES "Use 64-Bit indices. Reverts to 32-bit if turned off"
       ON)
option(HILBERT_ORDERING "Use Hilbert ordering instead of Morton" ON)
option(RUN_WEAK_SCALING "Run Entire Weak Scaling" ON)
option(ALLTOALLV_FIX "Use K-way all to all v" OFF)
option(SPLITTER_SELECTION_FIX "Turn on Splitter Selection fix" OFF)
option(DIM_2 "use the two dimentional sorting" OFF)
option(PROFILE_TREE_SORT
       "Profile the tree sort code, construction and balancing. " OFF)
option(WITH_BLAS_LAPACK "build using BLAS and LAPACk" ON)
option(MANUAL_BLAS_LAPACK "configure BLAS and LAPACK Manually" OFF)
option(DENDRO_VTK_BINARY "write vtk/vtu files in binary mode " ON)
option(
  DENDRO_VTK_ZLIB_COMPRES
  "write vtk/vtu files in binary mode with zlib compression (only compatible with binary mode) "
  OFF)
option(
  ALLTOALL_SPARSE
  "uses isend irecv for ghost exchange (for some mpi implementations this might be efficient)"
  OFF)
option(ENABLE_DENDRO_PROFILE_COUNTERS
       " enables dendro internal profile counters. " OFF)
option(
  RK_SOLVER_OVERLAP_COMM_AND_COMP
  " enables RK solver comm. & comp. overlap. (might be good in large scale runs) "
  ON)
option(WITH_CUDA " build dendro with cuda " OFF)
option(BUILD_WITH_PETSC " build dendro with PETSC " OFF)
option(USE_FD_INTERP_FOR_UNZIP "use FD style interpolation for unzip" OFF)
option(MPIX_CUDA_AWARE_SUPPORT "allow MPI to be aware of CUDA support" OFF)
option(DENDRO_ENABLE_GHOST_COMPRESSION
       "Enable compression on Ghost Communication" ON)

option(DVEC_ZERO_ALLOC "Make the memory allocation of dvec on CPU initialize as zeros" OFF)

option(DENDRO_USE_NEW_DERIVS "Use the new derivs" ON)


# option(KWAY "K parameter for alltoallv_kway" 128)
set(KWAY
    128
    CACHE STRING 128)
set(OCT2BLK_COARSEST_LEV
    0
    CACHE STRING 0)
set(NUM_NPES_THRESHOLD
    2
    CACHE STRING 2)

# ghost compression needs to be defined outside dendrolib as well
if(DENDRO_ENABLE_GHOST_COMPRESSION)
  add_definitions(-DDENDRO_ENABLE_GHOST_COMPRESSION)
endif()

if (DENDRO_USE_NEW_DERIVS)
  add_definitions(-DDENDRO_USE_NEW_DERIVS)
  set(DVEC_ZERO_ALLOC ON)
endif()

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# then the build type from CMAKE
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      "Release"
      CACHE STRING "Choose the type of build, options are: Debug Release "
            FORCE)
endif()

# ==========================
# == CUDA PACKAGE FINDING ==
# ==========================
if(WITH_CUDA)
  set(OCT2BLK_COARSEST_LEV
      31
      CACHE STRING 31)
  add_definitions(-DOCT2BLK_COARSEST_LEV=${OCT2BLK_COARSEST_LEV})
  enable_language(CUDA)
  add_definitions(-DWITH_CUDA)

  if(MPIX_CUDA_AWARE_SUPPORT)
    add_definitions(-DMPIX_CUDA_AWARE_SUPPORT)
  endif()
endif()
add_definitions(-DOCT2BLK_COARSEST_LEV=${OCT2BLK_COARSEST_LEV})

# =============================
# == BLAS and LAPACK FINDING ==
# =============================
if(MANUAL_BLAS_LAPACK)
  if("$ENV{BLAS}" STREQUAL "" OR "$ENV{LAPACK}" STREQUAL "")
    message(FATAL_ERROR "MANUAL_BLAS_LAPACK is ON, but BLAS/LAPACK env vars are empty.")
  endif()

  message(STATUS "Configuring BLAS/LAPACK Manually...")

  set(LAPACKE_DIR "$ENV{LAPACK}/LAPACKE")
  set(BLAS_LIBS "$ENV{BLAS}/lib")
  set(LAPACK_LIBS "$ENV{LAPACK}/lib")

  set(MANUAL_LIBS
        "${LAPACK_LIBS}/liblapacke.a"
        "${LAPACK_LIBS}/liblapack.a"
        "${BLAS_LIBS}/libblas.a"
        "-static" "libgfortran.a" "libquadmath.a"
    )

  if(NOT TARGET BLAS::BLAS)
    add_library(BLAS::BLAS INTERFACE IMPORTED)
    set_target_properties(BLAS::BLAS PROPERTIES INTERFACE_LINK_LIBRARIES "${MANUAL_LIBS}")
  endif()

  if(NOT TARGET LAPACK::LAPACK)
    add_library(LAPACK::LAPACK INTERFACE IMPORTED)
    set_target_properties(LAPACK::LAPACK PROPERTIES INTERFACE_LINK_LIBRARIES "${MANUAL_LIBS}")
  endif()

  # prevent dendrolib from trying to find_package and overwriting this
  set(BLAS_FOUND TRUE)
  set(LAPACK_FOUND TRUE)
endif()

if(WITH_BLAS_LAPACK)
  add_definitions(-DWITH_BLAS_LAPACK)
endif()

# ===================
# == PETSC FINDING ==
# ===================
if(BUILD_WITH_PETSC)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules")
  find_package(PETSc REQUIRED)
  add_definitions(-DBUILD_WITH_PETSC)

endif()

if(DIM_2)
  add_definitions(-DDIM_2)
endif()

if(PROFILE_TREE_SORT)
  add_definitions(-DPROFILE_TREE_SORT)
endif()

if(USE_64BIT_INDICES)
  add_definitions(-DUSE_64BIT_INDICES)
  # message('Configuring 64BIT indices')
endif()

if(HILBERT_ORDERING)
  add_definitions(-DHILBERT_ORDERING)
  # message (Configuring with HIlbert)
endif()

if(RUN_WEAK_SCALING)
  add_definitions(-DRUN_WEAK_SCALING)
  # message('Configuring with Weak Scaling')
endif()

if(ALLTOALLV_FIX)
  add_definitions(-DALLTOALLV_FIX)
  add_definitions(-DKWAY=${KWAY})
endif()

if(SPLITTER_SELECTION_FIX)
  add_definitions(-DSPLITTER_SELECTION_FIX)
  add_definitions(-DNUM_NPES_THRESHOLD=${NUM_NPES_THRESHOLD})
endif()

if(ALLTOALL_SPARSE)
  add_definitions(-DALLTOALL_SPARSE)
endif()

if(ENABLE_DENDRO_PROFILE_COUNTERS)
  add_definitions(-DENABLE_DENDRO_PROFILE_COUNTERS)
  add_definitions(-D__PROFILE_CTX__)
  add_definitions(-D__PROFILE_ETS__)
  add_definitions(-D__PROFILE_ENUTS__)
endif()

if(RK_SOLVER_OVERLAP_COMM_AND_COMP)
  add_definitions(-DRK_SOLVER_OVERLAP_COMM_AND_COMP)
endif()

if(DENDRO_VTK_BINARY)

else()
  set(DENDRO_VTK_ZLIB_COMPRES OFF)
endif()

if(DENDRO_VTK_BINARY)
  add_definitions(-DDENDRO_VTU_BINARY)
  if(DENDRO_VTK_ZLIB_COMPRES)
    add_definitions(-DDENDRO_VTU_ZLIB)
  endif()
else()
  add_definitions(-DDENDRO_VTU_ASCII)
endif()

if(USE_FD_INTERP_FOR_UNZIP)
  add_definitions(-DUSE_FD_INTERP_FOR_UNZIP)
endif()

# set(NUFFT_SRC ${CMAKE_CURRENT_SOURCE_DIR}/external/nufft3d-1.3.2/nufft3df90.f
# ${CMAKE_CURRENT_SOURCE_DIR}/external/nufft3d-1.3.2/dirft3d.f
# ${CMAKE_CURRENT_SOURCE_DIR}/external/nufft3d-1.3.2/dfftpack.f
# ${CMAKE_CURRENT_SOURCE_DIR}/external/nufft3d-1.3.2/next235.f )

# add_library(nufft3d_v1.3 STATIC ${NUFFT_SRC})

add_definitions(-DMATVEC_PROFILE)


if(DVEC_ZERO_ALLOC)
  add_definitions(-DDVEC_ZERO_ALLOC)
endif()

# === END REQUIRED CONFIGURATIONS FOR DENDRO ===

set(CHECK_GIT_INCLUDE_PATH ${CMAKE_SOURCE_DIR}/cmake-modules)
include(${CHECK_GIT_INCLUDE_PATH}/CheckGit.cmake)
message(STATUS "Setting up adding Git Information directly to compile options, header file should be in ${CHECK_GIT_INCLUDE_PATH}")
checkgitsetup()

include(FetchContent)
FetchContent_Declare(
  toml11
  GIT_REPOSITORY "https://github.com/ToruNiina/toml11"
  GIT_TAG "v4.4.0")
FetchContent_MakeAvailable(toml11)



# settable option for the user on where dendrolib is available
set(DENDRO_dendrolib_DIR "" CACHE PATH "Path to the local dendrolib source directory (optional)")
set(DENDRO_dendrolib_GIT_TAG "master" CACHE STRING "The Git tag, branch, or commit hash for dendrolib")

# now include the dendro repository for compiling the library
if (IS_DIRECTORY ${DENDRO_dendrolib_DIR})
  message(STATUS "Using user-provided dendrolib from: ${DENDRO_dendrolib_DIR}")
  add_subdirectory(${DENDRO_dendrolib_DIR} dendrolib)
  set(DENDRO_FETCHED FALSE)
else()
  message(STATUS "Dendrolib not found locally, fetching from repository...")
  FetchContent_Declare(
    dendrolib
    GIT_REPOSITORY "https://github.com/paralab/Dendro-5.01"
    GIT_TAG ${DENDRO_dendrolib_GIT_TAG}
  )

  FetchContent_MakeAvailable(dendrolib)

  set(DENDRO_dendrolib_DIR ${dendrolib_SOURCE_DIR} CACHE INTERNAL "Path to the fetched dendrolib source")

endif()

# then the subdirectory for the actual source of the project
add_subdirectory(solver)

