# # # # # # # ########################################################################
# CMAKE CONFIGURATION FILE FOR EM3 SOLVER
#
# This file in particular contains all of the options that apply specificially
# to the EM3 Solver binary as it is compiled.
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
option(SOLVER_PROFILE_HUMAN_READABLE "Enable human readable profile outputs" ON)
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

option(EM3_COMPUTE_ANALYTICAL
       "Adds the computation of the analytical solution *and* computes the diff"
       ON)

option(EM3_ENABLE_COMPACT_DERIVS
       "Swaps the RHS functionality to run compact finite differences" ON)

option(EM3_DEBUG_COMPACT_DERIVS
       "Enables some debug checks (like nan checking) in CFDs" ON)

option(
  EM3_USE_XSMM_MAT_MUL
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

if(SOLVER_ENABLE_CUDA)
  add_definitions(-DSOLVER_ENABLE_CUDA)
endif()

if(EM3_COMPUTE_ANALYTICAL)
  add_definitions(-DEM3_COMPUTE_ANALYTICAL)
endif()

if(EM3_ENABLE_COMPACT_DERIVS)
  add_definitions(-DEM3_ENABLE_COMPACT_DERIVS)
endif()

if(EM3_DEBUG_COMPACT_DERIVS)
  add_definitions(-DEM3_DEBUG_COMPACT_DERIVS)
endif()

if(EM3_USE_XSMM_MAT_MUL)
  add_definitions(-DEM3_USE_XSMM_MAT_MUL)
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
    ${CMAKE_SOURCE_DIR}/solver/include/compact_derivs.h)

set(CUSTOM_SOLVER_SRC
    src/debugger_tools.cpp
    src/rkSolver.cpp
    src/parameters.cpp
    src/grUtils.cpp
    src/rhs.cpp
    src/derivs.cpp
    src/physcon.cpp
    src/profile_params.cpp
    src/system_constraints.cpp
    src/dataUtils.cpp
    src/solverCtx.cpp
    src/compact_derivs.cpp)

set(SOURCE_FILES solver_main.cpp ${CUSTOM_SOLVER_INC} ${CUSTOM_SOLVER_SRC})
add_executable(em4Solver ${SOURCE_FILES})

# source directory is root of this project since this file is only called from
# the outer CMake call
target_include_directories(em4Solver PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(em4Solver PRIVATE ${CMAKE_SOURCE_DIR}/solver)
target_include_directories(em4Solver PRIVATE ${CMAKE_SOURCE_DIR}/solver/include)
target_include_directories(em4Solver PRIVATE ${CMAKE_SOURCE_DIR}/scripts)
target_include_directories(em4Solver PRIVATE ${DENDRO_dendrolib_SRC})
target_include_directories(em4Solver
                           PRIVATE ${DENDRO_dendrolib_SRC}/include)
target_include_directories(em4Solver
                           PRIVATE ${DENDRO_dendrolib_SRC}/test/include)
target_include_directories(
  em4Solver PRIVATE ${DENDRO_dendrolib_SRC}/examples/include)
target_include_directories(em4Solver
                           PRIVATE ${DENDRO_dendrolib_SRC}/ODE/include)
target_include_directories(em4Solver
                           PRIVATE ${DENDRO_dendrolib_SRC}/FEM/include)
target_include_directories(em4Solver
                           PRIVATE ${DENDRO_dendrolib_SRC}/LinAlg/include)
target_include_directories(em4Solver
                           PRIVATE ${DENDRO_dendrolib_SRC}/IO/include)
target_include_directories(em4Solver
                           PRIVATE ${DENDRO_dendrolib_SRC}/IO/vtk/include)
target_include_directories(em4Solver
                           PRIVATE ${DENDRO_dendrolib_SRC}/LinAlg/include)
target_include_directories(em4Solver
                           PRIVATE ${DENDRO_dendrolib_SRC}/IO/zlib/inc)

# and don't forget to include the MPI and GSL paths
target_include_directories(em4Solver PRIVATE ${MPI_INCLUDE_PATH})
target_include_directories(em4Solver PRIVATE ${GSL_INCLUDE_DIRS})

# also the toml library with the git submodule
target_include_directories(em4Solver PRIVATE ${EXTERNAL_LIB_HEADERS})

# then don't forget to link the libraries!!!!
target_link_libraries(em4Solver dendro5 ${LAPACK_LIBRARIES} ${MPI_LIBRARIES}
                      ${GSL_LIBRARIES} m toml11::toml11)

if(EM3_USE_XSMM_MAT_MUL)
  target_link_libraries(em4Solver xsmm)
endif()

set(SOURCE_FILES test_cfd_profiling.cpp src/derivs.cpp src/compact_derivs.cpp
                 include/derivs.h include/compact_derivs.h)
add_executable(testCFDS ${SOURCE_FILES})
target_include_directories(testCFDS PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(testCFDS PRIVATE ${CMAKE_SOURCE_DIR}/solver)
target_include_directories(testCFDS PRIVATE ${CMAKE_SOURCE_DIR}/solver/include)
target_include_directories(testCFDS PRIVATE ${DENDRO_dendrolib_SRC})
target_include_directories(testCFDS
                           PRIVATE ${DENDRO_dendrolib_SRC}/include)
target_include_directories(testCFDS
                           PRIVATE ${DENDRO_dendrolib_SRC}/test/include)
target_include_directories(
  testCFDS PRIVATE ${DENDRO_dendrolib_SRC}/examples/include)
target_include_directories(testCFDS
                           PRIVATE ${DENDRO_dendrolib_SRC}/ODE/include)
target_include_directories(testCFDS
                           PRIVATE ${DENDRO_dendrolib_SRC}/FEM/include)
target_include_directories(testCFDS
                           PRIVATE ${DENDRO_dendrolib_SRC}/LinAlg/include)
target_include_directories(testCFDS
                           PRIVATE ${DENDRO_dendrolib_SRC}/IO/include)
target_include_directories(testCFDS
                           PRIVATE ${DENDRO_dendrolib_SRC}/IO/vtk/include)
target_include_directories(testCFDS
                           PRIVATE ${DENDRO_dendrolib_SRC}/LinAlg/include)
target_include_directories(testCFDS
                           PRIVATE ${CMAKE_DOURCE_DIR}/dendrolib/IO/zlib/inc)

# also the toml library with the git submodule
target_include_directories(testCFDS PRIVATE ${EXTERNAL_LIB_HEADERS})
# then don't forget to link the libraries!!!!
target_link_libraries(testCFDS dendro5 ${LAPACK_LIBRARIES} ${MPI_LIBRARIES}
                      ${GSL_LIBRARIES} m toml11::toml11)

if(EM3_USE_XSMM_MAT_MUL)
  target_link_libraries(testCFDS xsmm)
endif()

# SOLVER FOR GLOBAL TESTS
set(SOURCE_FILES test_cfd_global.cpp src/derivs.cpp src/compact_derivs.cpp
                 include/derivs.h include/compact_derivs.h)
add_executable(cfdGlobal ${SOURCE_FILES})
target_include_directories(cfdGlobal PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(cfdGlobal PRIVATE ${CMAKE_SOURCE_DIR}/solver)
target_include_directories(cfdGlobal PRIVATE ${CMAKE_SOURCE_DIR}/solver/include)

# also the toml library with the git submodule
target_include_directories(cfdGlobal PRIVATE ${EXTERNAL_LIB_HEADERS})
# then don't forget to link the libraries!!!!
target_link_libraries(cfdGlobal dendro5 ${LAPACK_LIBRARIES} ${MPI_LIBRARIES}
                      ${GSL_LIBRARIES} m toml11::toml11)

if(EM3_USE_XSMM_MAT_MUL)
  target_link_libraries(cfdGlobal xsmm)
endif()

# SCALING SOURCES

set(SOURCE_FILES solver_scaling.cpp ${CUSTOM_SOLVER_INC} ${CUSTOM_SOLVER_SRC})
add_executable(solverScalingTest ${SOURCE_FILES})

target_include_directories(solverScalingTest PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(solverScalingTest PRIVATE ${CMAKE_SOURCE_DIR}/solver)
target_include_directories(solverScalingTest
                           PRIVATE ${CMAKE_SOURCE_DIR}/solver/include)
target_include_directories(solverScalingTest
                           PRIVATE ${CMAKE_SOURCE_DIR}/scripts)
target_include_directories(solverScalingTest
                           PRIVATE ${DENDRO_dendrolib_SRC})
target_include_directories(solverScalingTest
                           PRIVATE ${DENDRO_dendrolib_SRC}/include)
target_include_directories(solverScalingTest
                           PRIVATE ${DENDRO_dendrolib_SRC}/test/include)
target_include_directories(
  solverScalingTest PRIVATE ${DENDRO_dendrolib_SRC}/examples/include)
target_include_directories(solverScalingTest
                           PRIVATE ${DENDRO_dendrolib_SRC}/ODE/include)
target_include_directories(solverScalingTest
                           PRIVATE ${DENDRO_dendrolib_SRC}/FEM/include)
target_include_directories(solverScalingTest
                           PRIVATE ${DENDRO_dendrolib_SRC}/LinAlg/include)
target_include_directories(solverScalingTest
                           PRIVATE ${DENDRO_dendrolib_SRC}/IO/include)
target_include_directories(solverScalingTest
                           PRIVATE ${DENDRO_dendrolib_SRC}/IO/vtk/include)
target_include_directories(solverScalingTest
                           PRIVATE ${DENDRO_dendrolib_SRC}/LinAlg/include)
target_include_directories(solverScalingTest
                           PRIVATE ${DENDRO_dendrolib_SRC}/IO/zlib/inc)

# and don't forget to include the MPI and GSL paths
target_include_directories(solverScalingTest PRIVATE ${MPI_INCLUDE_PATH})
target_include_directories(solverScalingTest PRIVATE ${GSL_INCLUDE_DIRS})

# also the toml library with the git submodule
target_include_directories(solverScalingTest PRIVATE ${EXTERNAL_LIB_HEADERS})

# then don't forget to link the libraries!!!!
target_link_libraries(solverScalingTest dendro5 ${LAPACK_LIBRARIES}
                      ${MPI_LIBRARIES} ${GSL_LIBRARIES} m toml11::toml11)

if(EM3_USE_XSMM_MAT_MUL)
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
