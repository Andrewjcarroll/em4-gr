

#ifndef SFCSORTBENCH_PARAMETERS_H
#define SFCSORTBENCH_PARAMETERS_H

// library includes
#include <string.h>

#include <iostream>
// toml needs to be in the path (or included via submodule)
#include <memory>
#include <toml.hpp>

#include "derivatives.h"

// dendro only includes
#include "dendro.h"
#include "memory_pool.h"
#include "parUtils.h"

// project-specific includes
#include "compact_derivs.h"
#include "grDef.h"

// clang-format off
/*[[[cog

import cog
import dendrosym
import os

cog.outl('// clang-format on')

paramh_str, paramc_str = dendrosym.params.generate_all_parameter_text("solver", PARAM_SETUP_FILE)

cog.outl(paramh_str)

# when that's done, we also generate the sample file, can't think of another place to put this
with open(os.path.join(os.path.dirname(PARAM_SETUP_FILE), "solver_parameters.sample.toml"), "w") as f:
    f.write(dendrosym.params.generate_sample_config_file_text("solver", PARAM_SETUP_FILE))

]]]*/
// clang-format on
namespace dsolve {
void readParamFile(const char* inFile, MPI_Comm comm);
void dumpParamFile(std::ostream& sout, int root, MPI_Comm comm);

extern mem::memory_pool<double> SOLVER_MEM_POOL;

// extern RefinementMode SOLVER_REFINEMENT_MODE;

/**@brief SOLVER RK coordinate time*/
extern double SOLVER_CURRENT_RK_COORD_TIME;

/**@brief SOLVER RK step*/
extern unsigned int SOLVER_CURRENT_RK_STEP;

extern double* SOLVER_DERIV_WORKSPACE;
// number of derivatives, the greater between the RHS and Constraint
// TODO: this needs to be automated!!!!!!!!! ESPECIALLY WITH ADVANCED
// DERIVATIVES AS OF THIS MOMENT: THERE ARE 90 IN CONSTRAINTS AS OF THIS MOMENT:
// THERE ARE NOTE: KO DERIVATIVES ARE NOT INCLUDED IN THIS NUMBER
const unsigned int SOLVER_NUM_DERIVATIVES = 186;
}  // namespace dsolve

namespace dsolve {
/** @brief: Dendro version number, usually 5.0 especially for this project */
static const double DENDRO_VERSION = 5.0;

extern dendro_cfd::DerType SOLVER_DERIV_TYPE;
extern dendro_cfd::DerType2nd SOLVER_2ND_DERIV_TYPE;
/**@brief: Used to choose which compact finite difference deriv Filter to use*/
extern dendro_cfd::FilterType SOLVER_FILTER_TYPE;
extern unsigned int SOLVER_FILTER_FREQ;

extern dendro_cfd::BoundaryType SOLVER_DERIV_CLOSURE_TYPE;

extern unsigned int SOLVER_BL_DERIV_MAT_NUM;

extern double SOLVER_KIM_FILTER_KC;
extern double SOLVER_KIM_FILTER_EPS;

extern double EM4_NOISE_AMPLITUDE;

extern double EM4_ID_AMP1;
extern double EM4_ID_LAMBDA1;

extern double SOLVER_ETA_CONST;
extern double SOLVER_ETA_R0;
extern double SOLVER_ETA_DAMPING_EXP;

extern unsigned int SOLVER_PROFILE_OUTPUT_FREQ;

/** @brief: Element order for the computations */
extern unsigned int SOLVER_ELE_ORDER;

/** @brief: Padding width for each of the blocks */
extern unsigned int SOLVER_PADDING_WIDTH;

/** @brief: The number of total variables */
static const unsigned int SOLVER_NUM_VARS = 8;

/** @brief: Number of constraint variables */
static const unsigned int SOLVER_CONSTRAINT_NUM_VARS = 2;

/** @brief: Number of RK45 stages that should be performed */
static const unsigned int SOLVER_RK45_STAGES = 6;

/** @brief: Number of RK4 stages that should be performed */
static const unsigned int SOLVER_RK4_STAGES = 4;

/** @brief: Number of RK3 stages that should be performed */
static const unsigned int SOLVER_RK3_STAGES = 3;

/** @brief: Adaptive time step update safety factor */
static const double SOLVER_SAFETY_FAC = 0.8;

/** @brief: Number of internal variables */
static const unsigned int SOLVER_NUM_VARS_INTENL =
    (SOLVER_RK45_STAGES + 1) * SOLVER_NUM_VARS;

/** @brief: Minimum black hole domain, to be added to the parameter file for
 * running! */
extern double SOLVER_COMPD_MIN[3];

/** @brief: Maximum black hole domain, to be added to the parameter file for
 * running! */
extern double SOLVER_COMPD_MAX[3];

/** @brief: Minimum coordinates of the OCTREE */
extern double SOLVER_OCTREE_MIN[3];

/** @brief: Maximum coordinates of the OCTREE */
extern double SOLVER_OCTREE_MAX[3];

/** @brief: Output frequency for the solution, for saving to VTU file */
extern unsigned int SOLVER_IO_OUTPUT_FREQ;

/** @brief: Timestep output frequency */
extern unsigned int SOLVER_TIME_STEP_OUTPUT_FREQ;

extern unsigned int SOLVER_NUM_CONSOLE_OUTPUT_VARS;
extern unsigned int SOLVER_CONSOLE_OUTPUT_VARS[6];

extern unsigned int SOLVER_NUM_CONSOLE_OUTPUT_CONSTRAINTS;
extern unsigned int SOLVER_CONSOLE_OUTPUT_CONSTRAINTS[2];

/** @brief: Frequency for performing remeshing test based on wavelets */
extern unsigned int SOLVER_REMESH_TEST_FREQ;

/** @brief: Frequency for checkpoint saving */
extern unsigned int SOLVER_CHECKPT_FREQ;

/** @brief: Option for restoring from a checkpoint (will restore if set to 1) */
extern unsigned int SOLVER_RESTORE_SOLVER;

/** @brief: Disable AMR and enable block adaptivity */
extern unsigned int SOLVER_ENABLE_BLOCK_ADAPTIVITY;

/** @brief: File prefix for the VTU files that will be saved */
extern std::string SOLVER_VTU_FILE_PREFIX;

/** @brief: File prefix for the checkpoint files */
extern std::string SOLVER_CHKPT_FILE_PREFIX;

/** @brief: File prefix for the intermediate profile files */
extern std::string SOLVER_PROFILE_FILE_PREFIX;

/** @brief: Number variables for refinement */
extern unsigned int SOLVER_NUM_REFINE_VARS;

/** @brief: The IDs for the refinement variables, this will depend on the enum
 * that's generated from the Python */
extern unsigned int SOLVER_REFINE_VARIABLE_INDICES[8];

/** @brief: The number of evolution variables to put in the output of the files
 */
extern unsigned int SOLVER_NUM_EVOL_VARS_VTU_OUTPUT;

/** @brief: The number of constraint variables written to VTU files */
extern unsigned int SOLVER_NUM_CONST_VARS_VTU_OUTPUT;

/** @brief: Evolution variable IDs to be written to the VTU files */
extern unsigned int SOLVER_VTU_OUTPUT_EVOL_INDICES[8];

/** @brief: Constraint variable IDs to be written to the VTU files */
extern unsigned int SOLVER_VTU_OUTPUT_CONST_INDICES[2];

/** @brief: Solution output gap (instead of frequency, we can use to output the
 * solution if currentTime > lastIOOutputTime + SOLVER_IO_OUTPUT_GAP) */
extern unsigned int SOLVER_IO_OUTPUT_GAP;

/** @brief:  Grain size N/p, Where N number of total octants, p number of active
 * cores */
extern unsigned int SOLVER_DENDRO_GRAIN_SZ;

/** @brief: Dendro coarsening factor, if computed wavelet tol <
 * SOLVER_DENDRO_AMR_FAC*SOLVER_WAVELET_TOL */
extern double SOLVER_DENDRO_AMR_FAC;

/** @brief: Number of grid iterations untill the grid converges */
extern unsigned int SOLVER_INIT_GRID_ITER;

extern bool SOLVER_INIT_GRID_REINITIALIZE_EACH_TIME;

/** @brief: Splitter fix value */
extern unsigned int SOLVER_SPLIT_FIX;

/** @brief: The Courant factor: CFL stability number (specifies how
 * dt=SOLVER_CFL_FACTOR*dx) */
extern double SOLVER_CFL_FACTOR;

/** @brief: Simulation time begin */
extern unsigned int SOLVER_RK_TIME_BEGIN;

/** @brief: Simulation time end */
extern double SOLVER_RK_TIME_END;

/** @brief: RK method to use (0 -> RK3 , 1 -> RK4, 2 -> RK45) */
extern unsigned int SOLVER_RK_TYPE;

/** @brief: Prefered time step size (this is overwritten with the specified CFL
 * factor, not recommended to use this) */
extern double SOLVER_RK45_TIME_STEP_SIZE;

/** @brief: Desired tolerance value for the RK45 method (with adaptive time
 * stepping), NOT CURRENTLY USED */
extern double SOLVER_RK45_DESIRED_TOL;

/** @brief: The dissipation type to be used */
extern unsigned int DISSIPATION_TYPE;

/** @brief: The dissipation "NC", note this is only called in a comment for
 * "artificial dissipation" which appears to not be defined anywhere */
extern unsigned int SOLVER_DISSIPATION_NC;

/** @brief: The dissipation "S", note this is only called in a comment for
 * "artificial dissipation" which appears to not be defined anywhere */
extern unsigned int SOLVER_DISSIPATION_S;

/** @brief: The TS offset for LTS in SOLVER */
extern unsigned int SOLVER_LTS_TS_OFFSET;

/** @brief: Whether to output only the z slice in the VTU file */
extern bool SOLVER_VTU_Z_SLICE_ONLY;

/** @brief: Variable group size for the asynchronous unzip operation. This is an
 * async communication. (Upper bound should be SOLVER_NUM_VARS) */
extern unsigned int SOLVER_ASYNC_COMM_K;

/** @brief: Dendro load imbalance tolerance for flexible partitioning */
extern double SOLVER_LOAD_IMB_TOL;

/** @brief: Dimensionality of the octree, (meshing is supported only for 3D) */
extern unsigned int SOLVER_DIM;

/** @brief: Maximum and minimum levels of refinement of the mesh */
extern unsigned int SOLVER_MAXDEPTH;

extern unsigned int SOLVER_MINDEPTH;

/** @brief: Wavelet tolerance */
extern double SOLVER_WAVELET_TOL;

/** @brief: Set wavelet tolerance using a function (default 0) */
extern unsigned int SOLVER_USE_WAVELET_TOL_FUNCTION;

/** @brief: The maximum value of the wavelet tolerance */
extern double SOLVER_WAVELET_TOL_MAX;

/** @brief: Radius R0 for the wavelet tolerance function */
extern double SOLVER_WAVELET_TOL_FUNCTION_R0;

/** @brief: Radius R1 for the wavelet tolerance function */
extern double SOLVER_WAVELET_TOL_FUNCTION_R1;

/** @brief: Fd intergrid transfer enable or disable */
extern bool SOLVER_USE_FD_GRID_TRANSFER;

/** @brief: Refinement mode: 0 -> WAMR , 1 -> EH, 2 -> EH_WAMR 3 -> BH_loc based
 */
extern RefinementMode SOLVER_REFINEMENT_MODE;

extern double SOLVER_BLK_MIN_X;

extern double SOLVER_BLK_MIN_Y;

extern double SOLVER_BLK_MIN_Z;

extern double SOLVER_BLK_MAX_X;

extern double SOLVER_BLK_MAX_Y;

extern double SOLVER_BLK_MAX_Z;

extern double KO_DISS_SIGMA;

extern unsigned int SOLVER_ID_TYPE;

extern double SOLVER_GRID_MIN_X;

extern double SOLVER_GRID_MAX_X;

extern double SOLVER_GRID_MIN_Y;

extern double SOLVER_GRID_MAX_Y;

extern double SOLVER_GRID_MIN_Z;

extern double SOLVER_GRID_MAX_Z;

extern std::unique_ptr<dendroderivs::DendroDerivatives> SOLVER_DERIVS;

extern std::string SOLVER_DERIVTYPE_FIRST;
extern std::string SOLVER_DERIVTYPE_SECOND;
extern std::vector<double> SOLVER_DERIV_FIRST_COEFFS;
extern std::vector<double> SOLVER_DERIV_SECOND_COEFFS;
extern unsigned int SOLVER_DERIV_FIRST_MATID;
extern unsigned int SOLVER_DERIV_SECOND_MATID;

}  // namespace dsolve

//[[[end]]]

#endif  // SFCSORTBENCH_PARAMETERS_H
