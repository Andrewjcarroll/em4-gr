# ##############################################################################
# CMAKE CONFIGURATION FILE FOR ENTIRE PROJECT
#
# This file in particular contains all of the options that apply to all aspects
# of the program, including some Dendro options that are used when including the
# library.
#
# Note that all options and compiler flags set here will propagate to other
# directories such as src/ and dendro/
# ##############################################################################

# NOTE: cmake minimum is now set to 3.1
cmake_minimum_required(VERSION 3.1)

# full project name and the languages used
project(DendroEM3 C CXX Fortran)

# the standard we're using is c++11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

# set up fsanatize flags to check for memory leaks, need to set them for C++, X,
# and linker but only on debug
set(CMAKE_CXX_FLAGS_DEBUG
    "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address,undefined -fno-sanitize-recover"
)
set(CMAKE_C_FLAGS_DEBUG
    "${CMAKE_C_FLAGS_DEBUG} -fsanitize=address,undefined -fno-sanitize-recover")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG
    "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -fsanitize=address,undefined -fno-sanitize-recover"
)

# find OpenMP, MPI, and GSL
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(GSL REQUIRED)

# For now we just make it compulsory to have LAPACK installed. Later we will
# make it possible if LAPACK is not present to automaticall install before
# compiling dendro5
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS
      "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if(MPI_COMPILE_FLAGS)
  set(COMPILE_FLAGS "${COMPILE_FLAGS} ${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set(LINK_FLAGS "${LINK_FLAGS} ${MPI_LINK_FLAGS}")
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
  find_package(CUDA REQUIRED)
  if(CUDA_FOUND)
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    set(CUDA_SEPARABLE_COMPILATION ON)

    # list( APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30) list(
    # APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_50) list( APPEND
    # CUDA_NVCC_FLAGS -gencode arch=compute_60,code=sm_60) list( APPEND
    # CUDA_NVCC_FLAGS --generate-line-info) list( APPEND CUDA_NVCC_FLAGS
    # -gencode arch=compute_61,code=sm_61)

    # list( APPEND CUDA_NVCC_FLAGS -use_fast_math ) list( APPEND CUDA_NVCC_FLAGS
    # -maxrregcount 64 )
    list(
      APPEND
      CUDA_NVCC_FLAGS
      -std=c++11
      --ptxas-options=-O3
      --ptxas-options=-v
      -w
      -lineinfo)
    add_definitions(-DWITH_CUDA)

  endif()
  if(MPIX_CUDA_AWARE_SUPPORT)
    add_definitions(-DMPIX_CUDA_AWARE_SUPPORT)
  endif()
endif()
add_definitions(-DOCT2BLK_COARSEST_LEV=${OCT2BLK_COARSEST_LEV})

# ===== == BLAS and LAPAC FINDING ==
# ============================
if(WITH_BLAS_LAPACK)
  add_definitions(-DWITH_BLAS_LAPACK)
  if(MANUAL_BLAS_LAPACK)
    if("$ENV{BLAS}" STREQUAL "")
      message(
        "Environment Variable BLAS is not set. Please set it to BLAS directory")
    endif()

    if("$ENV{LAPACK}" STREQUAL "")
      message(
        "Enviroment Variable LAPACK is note set. Please set it to LAPACK directory. "
      )
    endif()
    set(LAPACKE_DIR $ENV{LAPACK}/LAPACKE)
    set(BLAS_LIBS $ENV{BLAS}/lib)
    set(LAPACK_LIBS $ENV{LAPACK}/lib)
    set(LAPACK_LINKER_FLAGS -llapacke -llapack -lblas -lgfortran -lquadmath)
    set(LAPACK_LIBRARIES
        ${LAPACK_LIBS}/liblapacke.a ${LAPACK_LIBS}/liblapack.a
        ${BLAS_LIBS}/libblas.a -static libgfortran.a libquadmath.a)
    set(LINK_FLAGS "${LINK_FLAGS} ${LAPACK_LINKER_FLAGS}")
  else()
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    set(LAPACK_LINKER_FLAGS -llapacke -llapack -lblas -lgfortran -lquadmath)
    set(LAPACKE_DIR $ENV{LAPACK}/LAPACKE)
    set(LINK_FLAGS "${LINK_FLAGS} ${LAPACK_LINKER_FLAGS}")
    # LAPACKE is not used anymore. find_library(LAPACKE_LIB NAMES lapacke
    # lapackelib liblapacke HINTS "/usr/lib/" )
    set(LAPACK_LIBRARIES ${LAPACK_LIBRARIES} ${LAPACKE_LIB})
    message(STATUS ${LAPACK_LIBRARIES})

    if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
      message(
        "${CMAKE_CXX_COMPILER_ID} compiler detected adding -mkl flag for BLAS LAPACK"
      )
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mkl")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mkl")
    endif()

  endif()

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

include(FetchContent)
FetchContent_Declare(
  toml11
  GIT_REPOSITORY "https://github.com/ToruNiina/toml11"
  GIT_TAG "v4.2.0")
FetchContent_MakeAvailable(toml11)



add_subdirectory(lib)

# add_subdirectory(dendrolib)
# set(DENDRO_dendrolib_DIR dendrolib)

# # TEMP: this links things to the development versions of dendrolib
add_subdirectory(../dendrolib_dfvk dendrolib)
set(DENDRO_dendrolib_DIR ../dendrolib_dfvk)

# then the subdirectory for the actual source of the project
add_subdirectory(solver)
