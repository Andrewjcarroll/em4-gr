# # # # # # # ########################################################################
# CMAKE CONFIGURATION FILE FOR EM4 SOLVER
#
# This file in particular contains all of the options that apply specificially
# to the EM4 Solver binary as it is compiled.
#
# Note that all options and compiler flags set in the ../CMakeLists.txt file
# (the one in the root directory of the project) *are carried through* here.
# Meaning, at this point, the C++ standard is set, OpenMP, MPI, GSL, and other
# libraries are found and linked. Also, general Dendro options, such as ordering
# options, dimensionality, CMake's build type, and others will also propagate
# through to here.
# # # # # # # ########################################################################

# The solver now requires the toml11 libary, now included in /lib this line
# tells CMAKE to link them
set(EXTERNAL_LIB_HEADERS ${CMAKE_SOURCE_DIR}/lib/toml11)

# ===========================
# == CUSTOM SOLVER SPECIFIC OPTIONS == These are options *specificaly* used for
# compiling and modifying the project
# ===========================
# option(COG_SOURCE_FILES "Enable cogging of source files" OFF)
option(SOLVER_PROFILE_HUMAN_READABLE "Enable human readable profile outputs" OFF)
option(
  SOLVER_CONSEC_COMM_SELECT
  "Rank selection is performed using consecutive ranks (works with an arbitrary number of processes). If off, the ranks are selected in binary tree fashion (only with powers of two processes)"
  ON)
option(SOLVER_ENABLE_VTU_OUTPUT "Enables VTU Output" ON)
option(SOLVER_COMPUTE_CONSTRAINTS "Enables constraint equation computation" ON)
option(SOLVER_ENABLE_VTU_CONSTRAINT_OUTPUT
       "Writes constraint output to VTU file" ON)
option(SOLVER_ETA_FUNCTION "Use function as ETA damping" OFF)
option(SOLVER_ENABLE_CUDA "Enable RHS computation with GPU acceleration" OFF)
option(EM4_ENABLE_PROFILING "Enable CPU/GPU solver profiling timers" ON)
option(SOLVER_USE_4TH_ORDER_DERIVS
       "Use 4th order derivative stencil calculations" OFF)
option(SOLVER_USE_6TH_ORDER_DERIVS
       "Use 6th order derivative stencil calculations" ON)
option(SOLVER_USE_8TH_ORDER_DERIVS
       "Use 8th order derivative stencil calculations" OFF)
option(
  SOLVER_ENABLE_AVX
  "Use vectorized computations for the RHS calculations (intel compilers ONLY)"
  OFF)
option(
  SOLVER_SAVE_RHS_EVERY_SINGLE_STEP
  "Allows you to save as VTU files all of the variables and their RHS at each time step"
  OFF)

option(EM4_COMPUTE_ANALYTICAL
       "Adds the computation of the analytical solution *and* computes the diff"
       ON)

option(EM4_ENABLE_COMPACT_DERIVS
       "Swaps the RHS functionality to run compact finite differences" OFF)

option(EM4_DEBUG_COMPACT_DERIVS
       "Enables some debug checks (like nan checking) in CFDs" OFF)

option(
  EM4_USE_XSMM_MAT_MUL
  "Enables the use of XSMM matrix multiplication (requires support in dendrolib)"
  OFF)

option(
  SOLVER_ENABLE_MERGED_BLOCKS
  "Allows the Compact Finite Differences to use merged blocks (requires OCT2BLK to not be 31)"
  OFF)

option(SOLVER_PROFILE_SCALING_RUN
       "Parallel scalabiltity test run (disable IO output)" OFF)

if(SOLVER_PROFILE_SCALING_RUN)
  add_definitions(-DSOLVER_PROFILE_SCALING_RUN)
  add_definitions(-D__PROFILE_ETS__)
  add_definitions(-D__PROFILE_CTX__)
endif(SOLVER_PROFILE_SCALING_RUN)

# setup for Intel-based vectorized computations
if(SOLVER_ENABLE_AVX)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    add_definitions(-DSOLVER_ENABLE_AVX)
  else()
    message(
      "WARNING: AVX vectorization not enabled due to detection of non-Intel compiler. Use `export CC=icc` and `export CXX=icpc` to use an Intel compiler."
    )
  endif()
endif(SOLVER_ENABLE_AVX)

# NOTE: adjustments to the derivative orders based on priority. If 4th order is
# enabled, it'll ignore the settings for later options and so on
if(SOLVER_USE_4TH_ORDER_DERIVS)
  set(SOLVER_USE_6TH_ORDER_DERIVS OFF)
  set(SOLVER_USE_8TH_ORDER_DERIVS OFF)
elseif(SOLVER_USE_6TH_ORDER_DERIVS)
  set(SOLVER_USE_4TH_ORDER_DERIVS OFF)
  set(SOLVER_USE_8TH_ORDER_DERIVS OFF)
elseif(SOLVER_USE_8TH_ORDER_DERIVS)
  set(SOLVER_USE_4TH_ORDER_DERIVS OFF)
  set(SOLVER_USE_6TH_ORDER_DERIVS OFF)
endif()

if(DENDRO_USE_NEW_DERIVS)
  # make sure the 6th order are set when using new derivs, as a default for definitions
  message(STATUS "NOTE that Dendro's 'New' Derivative system was enabled. By default, 6th order derivatives are 'configured', but should not be used.")
  set(SOLVER_USE_4TH_ORDER_DERIVS OFF)
  set(SOLVER_USE_6TH_ORDER_DERIVS ON)
  set(SOLVER_USE_8TH_ORDER_DERIVS OFF)
endif()


# now add the definitions for the compiler so that these options can be accessed
# by the code
if(SOLVER_USE_4TH_ORDER_DERIVS)
  add_definitions(-DSOLVER_USE_4TH_ORDER_DERIVS)
endif()

if(SOLVER_USE_6TH_ORDER_DERIVS)
  add_definitions(-DSOLVER_USE_6TH_ORDER_DERIVS)
endif()

if(SOLVER_USE_8TH_ORDER_DERIVS)
  add_definitions(-DSOLVER_USE_8TH_ORDER_DERIVS)
endif()

if(SOLVER_PROFILE_HUMAN_READABLE)
  add_definitions(-DSOLVER_PROFILE_HUMAN_READABLE)
endif()

if(SOLVER_CONSEC_COMM_SELECT)
  add_definitions(-DSOLVER_CONSEC_COMM_SELECT)
endif()

if(SOLVER_ENABLE_VTU_OUTPUT)
  add_definitions(-DSOLVER_ENABLE_VTU_OUTPUT)
endif()

if(SOLVER_COMPUTE_CONSTRAINTS)
  add_definitions(-DSOLVER_COMPUTE_CONSTRAINTS)
endif()

if(SOLVER_ENABLE_VTU_CONSTRAINT_OUTPUT)
  add_definitions(-DSOLVER_ENABLE_VTU_CONSTRAINT_OUTPUT)
endif()

# if(SOLVER_ENABLE_CUDA)
#   add_definitions(-DSOLVER_ENABLE_CUDA)
# endif()

if(EM4_ENABLE_PROFILING)
  add_definitions(-DEM4_ENABLE_PROFILING)
  add_definitions(-D__PROFILE_CTX__)
  add_definitions(-D__PROFILE_ETS__)
endif()

if(EM4_COMPUTE_ANALYTICAL)
  add_definitions(-DEM4_COMPUTE_ANALYTICAL)
endif()

if(EM4_ENABLE_COMPACT_DERIVS)
  add_definitions(-DEM4_ENABLE_COMPACT_DERIVS)
endif()

if(EM4_DEBUG_COMPACT_DERIVS)
  add_definitions(-DEM4_DEBUG_COMPACT_DERIVS)
endif()

if(EM4_USE_XSMM_MAT_MUL)
  add_definitions(-DEM4_USE_XSMM_MAT_MUL)
endif()

if(SOLVER_ENABLE_MERGED_BLOCKS)
  add_definitions(-DSOLVER_ENABLE_MERGED_BLOCKS)
endif()

set(CUSTOM_SOLVER_INC
    ${CMAKE_SOURCE_DIR}/solver/include/solver_main.h
    ${CMAKE_SOURCE_DIR}/solver/include/debugger_tools.h
    ${CMAKE_SOURCE_DIR}/solver/include/parameters.h
    ${CMAKE_SOURCE_DIR}/solver/include/grUtils.h
    ${CMAKE_SOURCE_DIR}/solver/include/grUtils.tcc
    ${CMAKE_SOURCE_DIR}/solver/include/rhs.h
    ${CMAKE_SOURCE_DIR}/solver/include/derivs.h
    ${CMAKE_SOURCE_DIR}/solver/include/physcon.h
    ${CMAKE_SOURCE_DIR}/solver/include/profile_params.h
    ${CMAKE_SOURCE_DIR}/solver/include/system_constraints.h
    ${CMAKE_SOURCE_DIR}/solver/include/dataUtils.h
    ${CMAKE_SOURCE_DIR}/solver/include/solverCtx.h
    )

set(SOLVER_SRC_FILES
    src/parameters.cpp
    src/grUtils.cpp
    src/rhs.cpp
    src/derivs.cpp
    src/solverCtx.cpp
    src/debugger_tools.cpp
    src/rkSolver.cpp
    src/physcon.cpp
    src/profile_params.cpp
    src/system_constraints.cpp
    src/dataUtils.cpp
)

set(SOLVER_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/scripts
)

if(DEFINED CHECK_GIT_INCLUDE_PATH)
  list(APPEND SOLVER_INCLUDE_DIRS ${CHECK_GIT_INCLUDE_PATH})
endif()

add_library(solver_common OBJECT ${SOLVER_SRC_FILES})
target_include_directories(solver_common PUBLIC ${SOLVER_INCLUDE_DIRS})
target_compile_options(solver_common PUBLIC -fext-numeric-literals)
target_link_libraries(solver_common PUBLIC dendro5 MPI::MPI_CXX GSL::gsl toml11::toml11 dendro_git_version_and_date)

if(WITH_CUDA)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=177")
  set(EM4_CUDA_SRC_FILES
        src/EM4CtxGPU.cu
    )
  target_sources(solver_common PRIVATE ${EM4_CUDA_SRC_FILES})
  set_target_properties(solver_common PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_include_directories(solver_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cuda/include)
  target_compile_definitions(solver_common PUBLIC SOLVER_ENABLE_CUDA)
endif()

macro(em4_add_executable TARGET_NAME)
  add_executable(${TARGET_NAME} ${ARGN})
  target_link_libraries(${TARGET_NAME} PRIVATE
        solver_common dendro5 MPI::MPI_CXX GSL::gsl toml11::toml11 m dendro_git_version_and_date)
  target_compile_options(${TARGET_NAME} PRIVATE -fext-numeric-literals)
endmacro()

if(WITH_CUDA)
  macro(em4_add_cuda_executable TARGET_NAME)
    add_executable(${TARGET_NAME} ${ARGN})
    set_target_properties(${TARGET_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_include_directories(${TARGET_NAME} PRIVATE ${SOLVER_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/cuda/include)

    target_compile_definitions(${TARGET_NAME} PRIVATE SOLVER_ENABLE_CUDA)

    target_link_libraries(${TARGET_NAME} PRIVATE
            solver_common dendro5 MPI::MPI_CXX GSL::gsl toml11::toml11 m dendro_git_version_and_date)
  endmacro()
endif()

em4_add_executable(em4Solver solver_main.cpp)

if(WITH_CUDA)
  em4_add_cuda_executable(em4SolverCUDA cuda/src/em4_cuda.cu)
endif()

# SCALING SOURCES
em4_add_executable(solverScalingTest solver_scaling.cpp)

if(EM4_USE_XSMM_MAT_MUL)
  target_link_libraries(solverScalingTest xsmm)
endif()


# then add the dependency add custom command to generate the source files from
# cog note that this doesn't actually *output* files, it actually will read
# existing source and update them if there are changes if(COG_SOURCE_FILES) #
# NOTE: the OUTPUT is set to rhs.cpp so that it'll always generate because #
# rhs.cpp is necessary for the project add_custom_command( OUTPUT
# ${CMAKE_SOURCE_DIR}/src/dummy.cpp DEPENDS
# ${CMAKE_SOURCE_DIR}/emdabssn_eqns_configs.py COMMAND python3 -m cogapp -D
# CONFIG_FILE_PATH=${CMAKE_SOURCE_DIR}/emdabssn_eqns_configs.py -D
# PARAM_SETUP_FILE=${CMAKE_SOURCE_DIR}/emdabssn_param_setup.toml -r
# @${CMAKE_SOURCE_DIR}/files_to_cog.txt COMMENT "Generating code based on the
# configurations in emda_eqns_configs.py. This may take a while as it
# optimizes." VERBATIM USES_TERMINAL WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# add_custom_target(COG_GENERATION DEPENDS ${CMAKE_SOURCE_DIR}/src/dummy.cpp)

# add_dependencies(em4Solver COG_GENERATION)

# endif()
