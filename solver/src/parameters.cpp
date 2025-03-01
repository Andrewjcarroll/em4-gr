
#include "parameters.h"

#include "compact_derivs.h"
#include "derivatives.h"
#include "parUtils.h"

/**
 * Global parameters used across the program
 *
 * NOTE: this will be generated via Python scripts in the future
 *
 */

// clang-format off
/*[[[cog

import cog
import dendrosym
cog.outl('// clang-format on')

paramh_str, paramc_str = dendrosym.params.generate_all_parameter_text("solver", PARAM_SETUP_FILE)

cog.outl(paramc_str)

]]]*/
// clang-format on
namespace dsolve {
mem::memory_pool<double> SOLVER_MEM_POOL = mem::memory_pool<double>(0, 16);
// RefinementMode SOLVER_REFINEMENT_MODE = RefinementMode::WAMR;
Point SOLVER_BH_LOC[2];
}  // namespace dsolve

namespace dsolve {

dendro_cfd::DerType SOLVER_DERIV_TYPE = dendro_cfd::DerType::CFD_NONE;
dendro_cfd::DerType2nd SOLVER_2ND_DERIV_TYPE =
    dendro_cfd::DerType2nd::CFD2ND_NONE;
dendro_cfd::FilterType SOLVER_FILTER_TYPE =
    dendro_cfd::FilterType::FILT_KO_DISS;

dendro_cfd::BoundaryType SOLVER_DERIV_CLOSURE_TYPE =
    dendro_cfd::BoundaryType::BLOCK_CFD_CLOSURE;

unsigned int SOLVER_ENABLE_BLOCK_ADAPTIVITY = 0;

unsigned int SOLVER_FILTER_FREQ = 10;

unsigned int SOLVER_BL_DERIV_MAT_NUM = 0;

double SOLVER_KIM_FILTER_KC = 0.88 * M_PI;
double SOLVER_KIM_FILTER_EPS = 0.25;

double EM4_NOISE_AMPLITUDE = 0.0;
double EM4_ID_AMP1 = 5.0;
double EM4_ID_LAMBDA1 = 0.05;

double SOLVER_ETA_CONST = 2.0;
double SOLVER_ETA_DAMPING_EXP = 2.0;
double SOLVER_ETA_R0 = 30.0;

unsigned int SOLVER_PROFILE_OUTPUT_FREQ = 1;

unsigned int SOLVER_ELE_ORDER = 6;
unsigned int SOLVER_PADDING_WIDTH = SOLVER_ELE_ORDER >> 1u;
double SOLVER_COMPD_MIN[3] = {-50.0, -50.0, -50.0};
double SOLVER_COMPD_MAX[3] = {50.0, 50.0, 50.0};
double SOLVER_OCTREE_MIN[3] = {0.0, 0.0, 0.0};
double SOLVER_OCTREE_MAX[3] = {(double)(1u << SOLVER_MAXDEPTH),
                               (double)(1u << SOLVER_MAXDEPTH),
                               (double)(1u << SOLVER_MAXDEPTH)};
unsigned int SOLVER_IO_OUTPUT_FREQ = 1000;
unsigned int SOLVER_TIME_STEP_OUTPUT_FREQ = 10;
unsigned int SOLVER_NUM_CONSOLE_OUTPUT_VARS = 6;
unsigned int SOLVER_CONSOLE_OUTPUT_VARS[6] = {0, 1, 2, 3, 4, 5};
unsigned int SOLVER_NUM_CONSOLE_OUTPUT_CONSTRAINTS = 2;
unsigned int SOLVER_CONSOLE_OUTPUT_CONSTRAINTS[2] = {0, 1};

unsigned int SOLVER_REMESH_TEST_FREQ = 10;
unsigned int SOLVER_CHECKPT_FREQ = 5000;
unsigned int SOLVER_RESTORE_SOLVER = 0;
std::string SOLVER_VTU_FILE_PREFIX = "vtu/solver_gr";
std::string SOLVER_CHKPT_FILE_PREFIX = "cp/solver_cp";
std::string SOLVER_PROFILE_FILE_PREFIX = "solver_prof";
unsigned int SOLVER_NUM_REFINE_VARS = 8;
unsigned int SOLVER_REFINE_VARIABLE_INDICES[8] = {0, 1, 2, 3, 4, 5,6,7};
unsigned int SOLVER_NUM_EVOL_VARS_VTU_OUTPUT = 14;
unsigned int SOLVER_NUM_CONST_VARS_VTU_OUTPUT = 1;
unsigned int SOLVER_VTU_OUTPUT_EVOL_INDICES[8] = {0, 1, 2, 3, 4, 5,6,7};
unsigned int SOLVER_VTU_OUTPUT_CONST_INDICES[2] = {0, 1};
unsigned int SOLVER_IO_OUTPUT_GAP = 1;
unsigned int SOLVER_DENDRO_GRAIN_SZ = 50;
double SOLVER_DENDRO_AMR_FAC = 0.1;
unsigned int SOLVER_INIT_GRID_ITER = 10;
bool SOLVER_INIT_GRID_REINITIALIZE_EACH_TIME = true;
unsigned int SOLVER_SPLIT_FIX = 2;
double SOLVER_CFL_FACTOR = 0.25;
unsigned int SOLVER_RK_TIME_BEGIN = 0;
double SOLVER_RK_TIME_END = 800;
unsigned int SOLVER_RK_TYPE = 1;
double SOLVER_RK45_TIME_STEP_SIZE = 0.01;
double SOLVER_RK45_DESIRED_TOL = 0.001;
unsigned int DISSIPATION_TYPE = 0;
unsigned int SOLVER_DISSIPATION_NC = 0;
unsigned int SOLVER_DISSIPATION_S = 0;
unsigned int SOLVER_LTS_TS_OFFSET = 0;
bool SOLVER_VTU_Z_SLICE_ONLY = true;
unsigned int SOLVER_ASYNC_COMM_K = 4;
double SOLVER_LOAD_IMB_TOL = 0.1;
unsigned int SOLVER_DIM = 3;
unsigned int SOLVER_MAXDEPTH = 16;
unsigned int SOLVER_MINDEPTH = 3;
double SOLVER_WAVELET_TOL = 1e-05;
unsigned int SOLVER_USE_WAVELET_TOL_FUNCTION = 3;
double SOLVER_WAVELET_TOL_MAX = 0.001;
double SOLVER_WAVELET_TOL_FUNCTION_R0 = 30.0;
double SOLVER_WAVELET_TOL_FUNCTION_R1 = 220.0;
bool SOLVER_USE_FD_GRID_TRANSFER = false;
RefinementMode SOLVER_REFINEMENT_MODE = static_cast<RefinementMode>(0);
double SOLVER_BLK_MIN_X = -6.0;
double SOLVER_BLK_MIN_Y = -6.0;
double SOLVER_BLK_MIN_Z = -1.0;
double SOLVER_BLK_MAX_X = 6.0;
double SOLVER_BLK_MAX_Y = 6.0;
double SOLVER_BLK_MAX_Z = 1.0;
double KO_DISS_SIGMA = 0.4;
unsigned int SOLVER_ID_TYPE = 0;
double SOLVER_GRID_MIN_X = -400.0;
double SOLVER_GRID_MAX_X = 400.0;
double SOLVER_GRID_MIN_Y = -400.0;
double SOLVER_GRID_MAX_Y = 400.0;
double SOLVER_GRID_MIN_Z = -400.0;
double SOLVER_GRID_MAX_Z = 400.0;

double SOLVER_CURRENT_RK_COORD_TIME = 0;
unsigned int SOLVER_CURRENT_RK_STEP = 0;

// NECESSARY ALLOCATION/START FOR DERIV WORKSPACE
double* SOLVER_DERIV_WORKSPACE = nullptr;

std::string SOLVER_DERIVTYPE_FIRST = "E6";
std::string SOLVER_DERIVTYPE_SECOND = "E6";
std::vector<double> SOLVER_DERIV_FIRST_COEFFS = {};
std::vector<double> SOLVER_DERIV_SECOND_COEFFS = {};
unsigned int SOLVER_DERIV_FIRST_MATID = 1;
unsigned int SOLVER_DERIV_SECOND_MATID = 1;

// default initialization
// this *MUST* be initialized
std::unique_ptr<dendroderivs::DendroDerivatives> SOLVER_DERIVS = nullptr;

}  // namespace dsolve
namespace dsolve {
void readParamFile(const char* inFile, MPI_Comm comm) {
    int rank, npes;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);

    auto file = toml::parse(inFile);

    if (file.contains("dsolve::EM4_NOISE_AMPLITUDE")) {
        dsolve::EM4_NOISE_AMPLITUDE =
            file["dsolve::EM4_NOISE_AMPLITUDE"].as_floating();
    }
    if (file.contains("dsolve::EM4_ID_AMP1")) {
        dsolve::EM4_ID_AMP1 = file["dsolve::EM4_ID_AMP1"].as_floating();
    }
    if (file.contains("dsolve::EM4_ID_LAMBDA1")) {
        dsolve::EM4_ID_LAMBDA1 = file["dsolve::EM4_ID_LAMBDA1"].as_floating();
    }
    if (file.contains("dsolve::SOLVER_ETA_CONST")) {
        dsolve::SOLVER_ETA_CONST =
            file["dsolve::SOLVER_ETA_CONST"].as_floating();
    }
    if (file.contains("dsolve::SOLVER_ETA_R0")) {
        dsolve::SOLVER_ETA_R0 = file["dsolve::SOLVER_ETA_R0"].as_floating();
    }
    if (file.contains("dsolve::SOLVER_ETA_DAMPING_EXP")) {
        dsolve::SOLVER_ETA_DAMPING_EXP =
            file["dsolve::SOLVER_ETA_DAMPING_EXP"].as_floating();
    }
    if (file.contains("dsolve::SOLVER_PROFILE_OUTPUT_FREQ")) {
        dsolve::SOLVER_PROFILE_OUTPUT_FREQ =
            file["dsolve::SOLVER_PROFILE_OUTPUT_FREQ"].as_integer();
    }

    if (file.contains("dsolve::SOLVER_DERIV_TYPE")) {
        SOLVER_DERIV_TYPE = static_cast<dendro_cfd::DerType>(
            file["dsolve::SOLVER_DERIV_TYPE"].as_integer());
    }
    if (file.contains("dsolve::SOLVER_2ND_DERIV_TYPE")) {
        SOLVER_2ND_DERIV_TYPE = static_cast<dendro_cfd::DerType2nd>(
            file["dsolve::SOLVER_2ND_DERIV_TYPE"].as_integer());
    }
    if (file.contains("dsolve::SOLVER_FILTER_TYPE")) {
        SOLVER_FILTER_TYPE = static_cast<dendro_cfd::FilterType>(
            file["dsolve::SOLVER_FILTER_TYPE"].as_integer());
    }

    if (file.contains("dsolve::SOLVER_DERIV_CLOSURE_TYPE")) {
        SOLVER_DERIV_CLOSURE_TYPE = static_cast<dendro_cfd::BoundaryType>(
            file["dsolve::SOLVER_DERIV_CLOSURE_TYPE"].as_integer());
    }

    if (file.contains("dsolve::SOLVER_BL_DERIV_MAT_NUM")) {
        dsolve::SOLVER_BL_DERIV_MAT_NUM =
            file["dsolve::SOLVER_BL_DERIV_MAT_NUM"].as_integer();
    }

    if (file.contains("dsolve::SOLVER_FILTER_FREQ")) {
        dsolve::SOLVER_FILTER_FREQ =
            file["dsolve::SOLVER_FILTER_FREQ"].as_integer();
    }

    if (file.contains("dsolve::SOLVER_KIM_FILTER_KC")) {
        dsolve::SOLVER_KIM_FILTER_KC =
            file["dsolve::SOLVER_KIM_FILTER_KC"].as_floating();
    }

    if (file.contains("dsolve::SOLVER_KIM_FILTER_EPS")) {
        dsolve::SOLVER_KIM_FILTER_EPS =
            file["dsolve::SOLVER_KIM_FILTER_EPS"].as_floating();
    }

    if (file.contains("dsolve::SOLVER_ELE_ORDER")) {
        dsolve::SOLVER_ELE_ORDER =
            file["dsolve::SOLVER_ELE_ORDER"].as_integer();
    }

    // padding width is half the element order
    // TODO: could potentially make it so element order is double, but
    // whatever
    dsolve::SOLVER_PADDING_WIDTH = dsolve::SOLVER_ELE_ORDER >> 1u;

    if (file.contains("dsolve::SOLVER_IO_OUTPUT_FREQ")) {
        dsolve::SOLVER_IO_OUTPUT_FREQ =
            file["dsolve::SOLVER_IO_OUTPUT_FREQ"].as_integer();
    }

    if (file.contains("dsolve::SOLVER_TIME_STEP_OUTPUT_FREQ")) {
        dsolve::SOLVER_TIME_STEP_OUTPUT_FREQ =
            file["dsolve::SOLVER_TIME_STEP_OUTPUT_FREQ"].as_integer();
    }

    if (file.contains("dsolve::SOLVER_NUM_CONSOLE_OUTPUT_VARS")) {
        dsolve::SOLVER_NUM_CONSOLE_OUTPUT_VARS =
            file["dsolve::SOLVER_NUM_CONSOLE_OUTPUT_VARS"].as_integer();
    }

    if (file.contains("dsolve::SOLVER_CONSOLE_OUTPUT_VARS")) {
        for (int i = 0; i < dsolve::SOLVER_NUM_CONSOLE_OUTPUT_VARS; ++i) {
            dsolve::SOLVER_CONSOLE_OUTPUT_VARS[i] =
                file["dsolve::SOLVER_CONSOLE_OUTPUT_VARS"][i].as_integer();
        }
    }

    if (file.contains("dsolve::SOLVER_NUM_CONSOLE_OUTPUT_CONSTRAINTS")) {
        dsolve::SOLVER_NUM_CONSOLE_OUTPUT_CONSTRAINTS =
            file["dsolve::SOLVER_NUM_CONSOLE_OUTPUT_CONSTRAINTS"].as_integer();
    }

    if (file.contains("dsolve::SOLVER_CONSOLE_OUTPUT_CONSTRAINTS")) {
        for (int i = 0; i < dsolve::SOLVER_NUM_CONSOLE_OUTPUT_CONSTRAINTS;
             ++i) {
            dsolve::SOLVER_CONSOLE_OUTPUT_CONSTRAINTS[i] =
                file["dsolve::SOLVER_CONSOLE_OUTPUT_CONSTRAINTS"][i]
                    .as_integer();
        }
    }

    if (file.contains("dsolve::SOLVER_REMESH_TEST_FREQ")) {
        dsolve::SOLVER_REMESH_TEST_FREQ =
            file["dsolve::SOLVER_REMESH_TEST_FREQ"].as_integer();
    }

    if (file.contains("dsolve::SOLVER_CHECKPT_FREQ")) {
        dsolve::SOLVER_CHECKPT_FREQ =
            file["dsolve::SOLVER_CHECKPT_FREQ"].as_integer();
    }

    if (file.contains("dsolve::SOLVER_RESTORE_SOLVER")) {
        dsolve::SOLVER_RESTORE_SOLVER =
            file["dsolve::SOLVER_RESTORE_SOLVER"].as_integer();
    }

    if (file.contains("dsolve::SOLVER_ENABLE_BLOCK_ADAPTIVITY")) {
        dsolve::SOLVER_ENABLE_BLOCK_ADAPTIVITY =
            file["dsolve::SOLVER_ENABLE_BLOCK_ADAPTIVITY"].as_integer();
    }

    if (file.contains("dsolve::SOLVER_VTU_FILE_PREFIX")) {
        dsolve::SOLVER_VTU_FILE_PREFIX =
            file["dsolve::SOLVER_VTU_FILE_PREFIX"].as_string();
    }

    if (file.contains("dsolve::SOLVER_CHKPT_FILE_PREFIX")) {
        dsolve::SOLVER_CHKPT_FILE_PREFIX =
            file["dsolve::SOLVER_CHKPT_FILE_PREFIX"].as_string();
    }

    if (file.contains("dsolve::SOLVER_PROFILE_FILE_PREFIX")) {
        dsolve::SOLVER_PROFILE_FILE_PREFIX =
            file["dsolve::SOLVER_PROFILE_FILE_PREFIX"].as_string();
    }

    if (file.contains("dsolve::SOLVER_NUM_REFINE_VARS")) {
        dsolve::SOLVER_NUM_REFINE_VARS =
            file["dsolve::SOLVER_NUM_REFINE_VARS"].as_integer();
    }

    if (file.contains("dsolve::SOLVER_REFINE_VARIABLE_INDICES")) {
        for (int i = 0; i < dsolve::SOLVER_NUM_REFINE_VARS; ++i) {
            dsolve::SOLVER_REFINE_VARIABLE_INDICES[i] =
                file["dsolve::SOLVER_REFINE_VARIABLE_INDICES"][i].as_integer();
        }
    }

    if (file.contains("dsolve::SOLVER_NUM_EVOL_VARS_VTU_OUTPUT")) {
        dsolve::SOLVER_NUM_EVOL_VARS_VTU_OUTPUT =
            file["dsolve::SOLVER_NUM_EVOL_VARS_VTU_OUTPUT"].as_integer();
    }

    if (file.contains("dsolve::SOLVER_NUM_CONST_VARS_VTU_OUTPUT")) {
        dsolve::SOLVER_NUM_CONST_VARS_VTU_OUTPUT =
            file["dsolve::SOLVER_NUM_CONST_VARS_VTU_OUTPUT"].as_integer();
    }

    if (file.contains("dsolve::SOLVER_VTU_OUTPUT_EVOL_INDICES")) {
        for (int i = 0; i < dsolve::SOLVER_NUM_EVOL_VARS_VTU_OUTPUT; ++i) {
            dsolve::SOLVER_VTU_OUTPUT_EVOL_INDICES[i] =
                file["dsolve::SOLVER_VTU_OUTPUT_EVOL_INDICES"][i].as_integer();
        }
    }

    if (file.contains("dsolve::SOLVER_VTU_OUTPUT_CONST_INDICES")) {
        for (int i = 0; i < dsolve::SOLVER_NUM_CONST_VARS_VTU_OUTPUT; ++i) {
            dsolve::SOLVER_VTU_OUTPUT_CONST_INDICES[i] =
                file["dsolve::SOLVER_VTU_OUTPUT_CONST_INDICES"][i].as_integer();
        }
    }

    if (file.contains("dsolve::SOLVER_IO_OUTPUT_GAP")) {
        dsolve::SOLVER_IO_OUTPUT_GAP =
            file["dsolve::SOLVER_IO_OUTPUT_GAP"].as_integer();
    }

    if (file.contains("dsolve::SOLVER_DENDRO_GRAIN_SZ")) {
        dsolve::SOLVER_DENDRO_GRAIN_SZ =
            file["dsolve::SOLVER_DENDRO_GRAIN_SZ"].as_integer();
    }

    if (file.contains("dsolve::SOLVER_DENDRO_AMR_FAC")) {
        if (0.0 > file["dsolve::SOLVER_DENDRO_AMR_FAC"].as_floating() ||
            0.2 < file["dsolve::SOLVER_DENDRO_AMR_FAC"].as_floating()) {
            std::cerr << R"(Invalid value for "dsolve::SOLVER_DENDRO_AMR_FAC")"
                      << std::endl;
            exit(-1);
        }

        dsolve::SOLVER_DENDRO_AMR_FAC =
            file["dsolve::SOLVER_DENDRO_AMR_FAC"].as_floating();
    }

    if (file.contains("dsolve::SOLVER_INIT_GRID_ITER")) {
        dsolve::SOLVER_INIT_GRID_ITER =
            file["dsolve::SOLVER_INIT_GRID_ITER"].as_integer();
    }

    if (file.contains("dsolve::SOLVER_INIT_GRID_REINITIALIZE_EACH_TIME")) {
        dsolve::SOLVER_INIT_GRID_REINITIALIZE_EACH_TIME =
            file["dsolve::SOLVER_INIT_GRID_REINITIALIZE_EACH_TIME"]
                .as_boolean();
    }

    if (file.contains("dsolve::SOLVER_SPLIT_FIX")) {
        dsolve::SOLVER_SPLIT_FIX =
            file["dsolve::SOLVER_SPLIT_FIX"].as_integer();
    }

    if (file.contains("dsolve::SOLVER_CFL_FACTOR")) {
        if (0.0 > file["dsolve::SOLVER_CFL_FACTOR"].as_floating() ||
            0.5 < file["dsolve::SOLVER_CFL_FACTOR"].as_floating()) {
            std::cerr << R"(Invalid value for "dsolve::SOLVER_CFL_FACTOR")"
                      << std::endl;
            exit(-1);
        }

        dsolve::SOLVER_CFL_FACTOR =
            file["dsolve::SOLVER_CFL_FACTOR"].as_floating();
    }

    if (file.contains("dsolve::SOLVER_RK_TIME_BEGIN")) {
        dsolve::SOLVER_RK_TIME_BEGIN =
            file["dsolve::SOLVER_RK_TIME_BEGIN"].as_integer();
    }

    if (file.contains("dsolve::SOLVER_RK_TIME_END")) {
        dsolve::SOLVER_RK_TIME_END =
            file["dsolve::SOLVER_RK_TIME_END"].as_floating();
    }

    if (file.contains("dsolve::SOLVER_RK_TYPE")) {
        dsolve::SOLVER_RK_TYPE = file["dsolve::SOLVER_RK_TYPE"].as_integer();
    }

    if (file.contains("dsolve::SOLVER_RK45_TIME_STEP_SIZE")) {
        if (0.0 > file["dsolve::SOLVER_RK45_TIME_STEP_SIZE"].as_floating() ||
            0.02 < file["dsolve::SOLVER_RK45_TIME_STEP_SIZE"].as_floating()) {
            std::cerr
                << R"(Invalid value for "dsolve::SOLVER_RK45_TIME_STEP_SIZE")"
                << std::endl;
            exit(-1);
        }

        dsolve::SOLVER_RK45_TIME_STEP_SIZE =
            file["dsolve::SOLVER_RK45_TIME_STEP_SIZE"].as_floating();
    }

    if (file.contains("dsolve::SOLVER_RK45_DESIRED_TOL")) {
        if (0.0 > file["dsolve::SOLVER_RK45_DESIRED_TOL"].as_floating() ||
            0.002 < file["dsolve::SOLVER_RK45_DESIRED_TOL"].as_floating()) {
            std::cerr
                << R"(Invalid value for "dsolve::SOLVER_RK45_DESIRED_TOL")"
                << std::endl;
            exit(-1);
        }

        dsolve::SOLVER_RK45_DESIRED_TOL =
            file["dsolve::SOLVER_RK45_DESIRED_TOL"].as_floating();
    }

    if (file.contains("dsolve::DISSIPATION_TYPE")) {
        dsolve::DISSIPATION_TYPE =
            file["dsolve::DISSIPATION_TYPE"].as_integer();
    }

    if (file.contains("dsolve::SOLVER_DISSIPATION_NC")) {
        dsolve::SOLVER_DISSIPATION_NC =
            file["dsolve::SOLVER_DISSIPATION_NC"].as_integer();
    }

    if (file.contains("dsolve::SOLVER_DISSIPATION_S")) {
        dsolve::SOLVER_DISSIPATION_S =
            file["dsolve::SOLVER_DISSIPATION_S"].as_integer();
    }

    if (file.contains("dsolve::SOLVER_LTS_TS_OFFSET")) {
        dsolve::SOLVER_LTS_TS_OFFSET =
            file["dsolve::SOLVER_LTS_TS_OFFSET"].as_integer();
    }

    if (file.contains("dsolve::SOLVER_VTU_Z_SLICE_ONLY")) {
        dsolve::SOLVER_VTU_Z_SLICE_ONLY =
            file["dsolve::SOLVER_VTU_Z_SLICE_ONLY"].as_boolean();
    }

    if (file.contains("dsolve::SOLVER_ASYNC_COMM_K")) {
        dsolve::SOLVER_ASYNC_COMM_K =
            file["dsolve::SOLVER_ASYNC_COMM_K"].as_integer();
    }

    if (file.contains("dsolve::SOLVER_LOAD_IMB_TOL")) {
        if (0.0 > file["dsolve::SOLVER_LOAD_IMB_TOL"].as_floating() ||
            0.2 < file["dsolve::SOLVER_LOAD_IMB_TOL"].as_floating()) {
            std::cerr << R"(Invalid value for "dsolve::SOLVER_LOAD_IMB_TOL")"
                      << std::endl;
            exit(-1);
        }

        dsolve::SOLVER_LOAD_IMB_TOL =
            file["dsolve::SOLVER_LOAD_IMB_TOL"].as_floating();
    }

    if (file.contains("dsolve::SOLVER_DIM")) {
        dsolve::SOLVER_DIM = file["dsolve::SOLVER_DIM"].as_integer();
    }

    if (file.contains("dsolve::SOLVER_MAXDEPTH")) {
        dsolve::SOLVER_MAXDEPTH = file["dsolve::SOLVER_MAXDEPTH"].as_integer();
    }

    if (file.contains("dsolve::SOLVER_MINDEPTH")) {
        dsolve::SOLVER_MINDEPTH = file["dsolve::SOLVER_MINDEPTH"].as_integer();
    }

    if (file.contains("dsolve::SOLVER_WAVELET_TOL")) {
        if (0.0 > file["dsolve::SOLVER_WAVELET_TOL"].as_floating() ||
            1e-04 < file["dsolve::SOLVER_WAVELET_TOL"].as_floating()) {
            std::cerr << R"(Invalid value for "dsolve::SOLVER_WAVELET_TOL")"
                      << std::endl;
            exit(-1);
        }

        dsolve::SOLVER_WAVELET_TOL =
            file["dsolve::SOLVER_WAVELET_TOL"].as_floating();
    }

    if (file.contains("dsolve::SOLVER_USE_WAVELET_TOL_FUNCTION")) {
        dsolve::SOLVER_USE_WAVELET_TOL_FUNCTION =
            file["dsolve::SOLVER_USE_WAVELET_TOL_FUNCTION"].as_integer();
    }

    if (file.contains("dsolve::SOLVER_WAVELET_TOL_MAX")) {
        if (0.0 > file["dsolve::SOLVER_WAVELET_TOL_MAX"].as_floating() ||
            0.002 < file["dsolve::SOLVER_WAVELET_TOL_MAX"].as_floating()) {
            std::cerr << R"(Invalid value for "dsolve::SOLVER_WAVELET_TOL_MAX")"
                      << std::endl;
            exit(-1);
        }

        dsolve::SOLVER_WAVELET_TOL_MAX =
            file["dsolve::SOLVER_WAVELET_TOL_MAX"].as_floating();
    }

    if (file.contains("dsolve::SOLVER_WAVELET_TOL_FUNCTION_R0")) {
        if (0.0 >
                file["dsolve::SOLVER_WAVELET_TOL_FUNCTION_R0"].as_floating() ||
            60.0 <
                file["dsolve::SOLVER_WAVELET_TOL_FUNCTION_R0"].as_floating()) {
            std::cerr
                << R"(Invalid value for "dsolve::SOLVER_WAVELET_TOL_FUNCTION_R0")"
                << std::endl;
            exit(-1);
        }

        dsolve::SOLVER_WAVELET_TOL_FUNCTION_R0 =
            file["dsolve::SOLVER_WAVELET_TOL_FUNCTION_R0"].as_floating();
    }

    if (file.contains("dsolve::SOLVER_WAVELET_TOL_FUNCTION_R1")) {
        if (0.0 >
                file["dsolve::SOLVER_WAVELET_TOL_FUNCTION_R1"].as_floating() ||
            440.0 <
                file["dsolve::SOLVER_WAVELET_TOL_FUNCTION_R1"].as_floating()) {
            std::cerr
                << R"(Invalid value for "dsolve::SOLVER_WAVELET_TOL_FUNCTION_R1")"
                << std::endl;
            exit(-1);
        }

        dsolve::SOLVER_WAVELET_TOL_FUNCTION_R1 =
            file["dsolve::SOLVER_WAVELET_TOL_FUNCTION_R1"].as_floating();
    }

    if (file.contains("dsolve::SOLVER_USE_FD_GRID_TRANSFER")) {
        dsolve::SOLVER_USE_FD_GRID_TRANSFER =
            file["dsolve::SOLVER_USE_FD_GRID_TRANSFER"].as_boolean();
    }

    if (file.contains("dsolve::SOLVER_REFINEMENT_MODE")) {
        if (0 > file["dsolve::SOLVER_REFINEMENT_MODE"].as_integer() ||
            (int)RefinementMode::REFINEMENT_END <
                file["dsolve::SOLVER_REFINEMENT_MODE"].as_integer()) {
            std::cerr << R"(Invalid value for "dsolve::SOLVER_REFINEMENT_MODE")"
                      << std::endl;
            exit(-1);
        }

        dsolve::SOLVER_REFINEMENT_MODE = static_cast<RefinementMode>(
            file["dsolve::SOLVER_REFINEMENT_MODE"].as_integer());
    }

    if (file.contains("dsolve::SOLVER_BLK_MIN_X")) {
        // if (-12.0 > file["dsolve::SOLVER_BLK_MIN_X"].as_floating() ||
        //     0.0 < file["dsolve::SOLVER_BLK_MIN_X"].as_floating()) {
        //     std::cerr << R"(Invalid value for
        //     "dsolve::SOLVER_BLK_MIN_X")"
        //               << std::endl;
        //     exit(-1);
        // }

        dsolve::SOLVER_BLK_MIN_X =
            file["dsolve::SOLVER_BLK_MIN_X"].as_floating();
    }

    if (file.contains("dsolve::SOLVER_BLK_MIN_Y")) {
        // if (-12.0 > file["dsolve::SOLVER_BLK_MIN_Y"].as_floating() ||
        //     0.0 < file["dsolve::SOLVER_BLK_MIN_Y"].as_floating()) {
        //     std::cerr << R"(Invalid value for
        //     "dsolve::SOLVER_BLK_MIN_Y")"
        //               << std::endl;
        //     exit(-1);
        // }

        dsolve::SOLVER_BLK_MIN_Y =
            file["dsolve::SOLVER_BLK_MIN_Y"].as_floating();
    }

    if (file.contains("dsolve::SOLVER_BLK_MIN_Z")) {
        // if (-12.0 > file["dsolve::SOLVER_BLK_MIN_Z"].as_floating() ||
        //     0.0 < file["dsolve::SOLVER_BLK_MIN_Z"].as_floating()) {
        //     std::cerr << R"(Invalid value for
        //     "dsolve::SOLVER_BLK_MIN_Z")"
        //               << std::endl;
        //     exit(-1);
        // }

        dsolve::SOLVER_BLK_MIN_Z =
            file["dsolve::SOLVER_BLK_MIN_Z"].as_floating();
    }

    if (file.contains("dsolve::SOLVER_BLK_MAX_X")) {
        // if (0.0 > file["dsolve::SOLVER_BLK_MAX_X"].as_floating() ||
        //     12.0 < file["dsolve::SOLVER_BLK_MAX_X"].as_floating()) {
        //     std::cerr << R"(Invalid value for
        //     "dsolve::SOLVER_BLK_MAX_X")"
        //               << std::endl;
        //     exit(-1);
        // }

        dsolve::SOLVER_BLK_MAX_X =
            file["dsolve::SOLVER_BLK_MAX_X"].as_floating();
    }

    if (file.contains("dsolve::SOLVER_BLK_MAX_Y")) {
        // if (0.0 > file["dsolve::SOLVER_BLK_MAX_Y"].as_floating() ||
        //     12.0 < file["dsolve::SOLVER_BLK_MAX_Y"].as_floating()) {
        //     std::cerr << R"(Invalid value for
        //     "dsolve::SOLVER_BLK_MAX_Y")"
        //               << std::endl;
        //     exit(-1);
        // }

        dsolve::SOLVER_BLK_MAX_Y =
            file["dsolve::SOLVER_BLK_MAX_Y"].as_floating();
    }

    if (file.contains("dsolve::SOLVER_BLK_MAX_Z")) {
        // if (0.0 > file["dsolve::SOLVER_BLK_MAX_Z"].as_floating() ||
        //     12.0 < file["dsolve::SOLVER_BLK_MAX_Z"].as_floating()) {
        //     std::cerr << R"(Invalid value for
        //     "dsolve::SOLVER_BLK_MAX_Z")"
        //               << std::endl;
        //     exit(-1);
        // }

        dsolve::SOLVER_BLK_MAX_Z =
            file["dsolve::SOLVER_BLK_MAX_Z"].as_floating();
    }

    if (file.contains("dsolve::KO_DISS_SIGMA")) {
        if (0.0 > file["dsolve::KO_DISS_SIGMA"].as_floating() ||
            0.8 < file["dsolve::KO_DISS_SIGMA"].as_floating()) {
            std::cerr << R"(Invalid value for "dsolve::KO_DISS_SIGMA")"
                      << std::endl;
            exit(-1);
        }

        dsolve::KO_DISS_SIGMA = file["dsolve::KO_DISS_SIGMA"].as_floating();
    }

    if (file.contains("dsolve::SOLVER_ID_TYPE")) {
        dsolve::SOLVER_ID_TYPE = file["dsolve::SOLVER_ID_TYPE"].as_integer();
    }

    if (file.contains("dsolve::SOLVER_GRID_MIN_X")) {
        if (-500.0 > file["dsolve::SOLVER_GRID_MIN_X"].as_floating() ||
            0.0 < file["dsolve::SOLVER_GRID_MIN_X"].as_floating()) {
            std::cerr << R"(Invalid value for "dsolve::SOLVER_GRID_MIN_X")"
                      << std::endl;
            exit(-1);
        }

        dsolve::SOLVER_GRID_MIN_X =
            file["dsolve::SOLVER_GRID_MIN_X"].as_floating();
    }

    if (file.contains("dsolve::SOLVER_GRID_MAX_X")) {
        if (0.0 > file["dsolve::SOLVER_GRID_MAX_X"].as_floating() ||
            500.0 < file["dsolve::SOLVER_GRID_MAX_X"].as_floating()) {
            std::cerr << R"(Invalid value for "dsolve::SOLVER_GRID_MAX_X")"
                      << std::endl;
            exit(-1);
        }

        dsolve::SOLVER_GRID_MAX_X =
            file["dsolve::SOLVER_GRID_MAX_X"].as_floating();
    }

    if (file.contains("dsolve::SOLVER_GRID_MIN_Y")) {
        if (-500.0 > file["dsolve::SOLVER_GRID_MIN_Y"].as_floating() ||
            0.0 < file["dsolve::SOLVER_GRID_MIN_Y"].as_floating()) {
            std::cerr << R"(Invalid value for "dsolve::SOLVER_GRID_MIN_Y")"
                      << std::endl;
            exit(-1);
        }

        dsolve::SOLVER_GRID_MIN_Y =
            file["dsolve::SOLVER_GRID_MIN_Y"].as_floating();
    }

    if (file.contains("dsolve::SOLVER_GRID_MAX_Y")) {
        if (0.0 > file["dsolve::SOLVER_GRID_MAX_Y"].as_floating() ||
            500.0 < file["dsolve::SOLVER_GRID_MAX_Y"].as_floating()) {
            std::cerr << R"(Invalid value for "dsolve::SOLVER_GRID_MAX_Y")"
                      << std::endl;
            exit(-1);
        }

        dsolve::SOLVER_GRID_MAX_Y =
            file["dsolve::SOLVER_GRID_MAX_Y"].as_floating();
    }

    if (file.contains("dsolve::SOLVER_GRID_MIN_Z")) {
        if (-500.0 > file["dsolve::SOLVER_GRID_MIN_Z"].as_floating() ||
            0.0 < file["dsolve::SOLVER_GRID_MIN_Z"].as_floating()) {
            std::cerr << R"(Invalid value for "dsolve::SOLVER_GRID_MIN_Z")"
                      << std::endl;
            exit(-1);
        }

        dsolve::SOLVER_GRID_MIN_Z =
            file["dsolve::SOLVER_GRID_MIN_Z"].as_floating();
    }

    if (file.contains("dsolve::SOLVER_GRID_MAX_Z")) {
        if (0.0 > file["dsolve::SOLVER_GRID_MAX_Z"].as_floating() ||
            500.0 < file["dsolve::SOLVER_GRID_MAX_Z"].as_floating()) {
            std::cerr << R"(Invalid value for "dsolve::SOLVER_GRID_MAX_Z")"
                      << std::endl;
            exit(-1);
        }

        dsolve::SOLVER_GRID_MAX_Z =
            file["dsolve::SOLVER_GRID_MAX_Z"].as_floating();
    }

    if (file.contains("SOLVER_DERIVTYPE_FIRST")) {
        SOLVER_DERIVTYPE_FIRST = file["SOLVER_DERIVTYPE_FIRST"].as_string();
    }

    if (file.contains("SOLVER_DERIVTYPE_SECOND")) {
        SOLVER_DERIVTYPE_SECOND = file["SOLVER_DERIVTYPE_SECOND"].as_string();
    }

    if (file.contains("SOLVER_DERIV_FIRST_COEFFS")) {
        SOLVER_DERIV_FIRST_COEFFS =
            toml::find<std::vector<double>>(file, "SOLVER_DERIV_FIRST_COEFFS");
    }

    if (file.contains("SOLVER_DERIV_SECOND_COEFFS")) {
        SOLVER_DERIV_SECOND_COEFFS =
            toml::find<std::vector<double>>(file, "SOLVER_DERIV_SECOND_COEFFS");
    }

    if (file.contains("SOLVER_DERIV_FIRST_MATID")) {
        SOLVER_DERIV_FIRST_MATID =
            file["SOLVER_DERIV_FIRST_MATID"].as_integer();
    }

    if (file.contains("SOLVER_DERIV_SECOND_MATID")) {
        SOLVER_DERIV_SECOND_MATID =
            file["SOLVER_DERIV_SECOND_MATID"].as_integer();
    }

    // COMPD_MIN and COMPD_MAX should be the same as the grid
    dsolve::SOLVER_COMPD_MIN[0] = dsolve::SOLVER_GRID_MIN_X;
    dsolve::SOLVER_COMPD_MIN[1] = dsolve::SOLVER_GRID_MIN_Y;
    dsolve::SOLVER_COMPD_MIN[2] = dsolve::SOLVER_GRID_MIN_Z;

    dsolve::SOLVER_COMPD_MAX[0] = dsolve::SOLVER_GRID_MAX_X;
    dsolve::SOLVER_COMPD_MAX[1] = dsolve::SOLVER_GRID_MAX_Y;
    dsolve::SOLVER_COMPD_MAX[2] = dsolve::SOLVER_GRID_MAX_Z;

    dsolve::SOLVER_OCTREE_MAX[0] = (double)(1u << dsolve::SOLVER_MAXDEPTH);
    dsolve::SOLVER_OCTREE_MAX[1] = (double)(1u << dsolve::SOLVER_MAXDEPTH);
    dsolve::SOLVER_OCTREE_MAX[2] = (double)(1u << dsolve::SOLVER_MAXDEPTH);

    std::cout << "ENABLE_BLOCK_ADAPTIVITY: "
              << dsolve::SOLVER_ENABLE_BLOCK_ADAPTIVITY << std::endl;

    dsolve::SOLVER_PADDING_WIDTH = dsolve::SOLVER_ELE_ORDER >> 1u;

    // establish the dendro derivatives class, this should always be built
    SOLVER_DERIVS = std::make_unique<dendroderivs::DendroDerivatives>(
        SOLVER_DERIVTYPE_FIRST, SOLVER_DERIVTYPE_SECOND, SOLVER_ELE_ORDER,
        SOLVER_DERIV_FIRST_COEFFS, SOLVER_DERIV_SECOND_COEFFS,
        SOLVER_DERIV_FIRST_MATID, SOLVER_DERIV_SECOND_MATID);

    // TODO: COMPD_MIN, COMPD_MAX should be GRID_MIN and GRID_MAX, not settable
    // by user

    // TODO: a lot of these parameters should come from the black hole portion
}

void dumpParamFile(std::ostream& sout, int root, MPI_Comm comm) {
    int rank, npes;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);

    if (rank == root) {
        sout << "Parameters read: " << std::endl;
        sout << "\tdsolve::DENDRO_VERSION: " << dsolve::DENDRO_VERSION
             << std::endl;

        sout << "\tdsolve::EM4_NOISE_AMPLITUDE: " << dsolve::EM4_NOISE_AMPLITUDE
             << std::endl;
        sout << "\tdsolve::EM4_ID_AMP1: " << dsolve::EM4_ID_AMP1 << std::endl;
        sout << "\tdsolve::EM4_ID_LAMBDA1: " << dsolve::EM4_ID_LAMBDA1
             << std::endl;

        // NOTE: the enum starts at -1, so we add one for the array
        sout << "\tdsolve::SOLVER_DERIV_TYPE: "
             << dendro_cfd::DER_TYPE_NAMES[dsolve::SOLVER_DERIV_TYPE + 1]
             << std::endl;
        sout
            << "\tdsolve::SOLVER_2ND_DERIV_TYPE: "
            << dendro_cfd::DER_TYPE_2ND_NAMES[dsolve::SOLVER_2ND_DERIV_TYPE + 1]
            << std::endl;
        sout << "\tdsolve::SOLVER_FILTER_TYPE: "
             << dendro_cfd::FILT_TYPE_NAMES[dsolve::SOLVER_FILTER_TYPE + 1]
             << std::endl;

        sout << "\tdsolve::SOLVER_FILTER_FREQ: " << dsolve::SOLVER_FILTER_FREQ
             << std::endl;

        sout << "\tdsolve::SOLVER_DERIV_CLOSURE_TYPE: "
             << dendro_cfd::BOUNDARY_TYPE_NAMES
                    [dsolve::SOLVER_DERIV_CLOSURE_TYPE]
             << std::endl;

        sout << "\tdsolve::SOLVER_BL_DERIV_MAT_NUM: "
             << dsolve::SOLVER_BL_DERIV_MAT_NUM << std::endl;

        sout << "\tdsolve::SOLVER_KIM_FILTER_KC: "
             << dsolve::SOLVER_KIM_FILTER_KC << std::endl;

        sout << "\tdsolve::SOLVER_KIM_FILTER_EPS: "
             << dsolve::SOLVER_KIM_FILTER_EPS << std::endl;

        sout << "\tdsolve::SOLVER_ELE_ORDER: " << dsolve::SOLVER_ELE_ORDER
             << std::endl;
        sout << "\tdsolve::SOLVER_PADDING_WIDTH: "
             << dsolve::SOLVER_PADDING_WIDTH << std::endl;
        sout << "\tdsolve::SOLVER_NUM_VARS: " << dsolve::SOLVER_NUM_VARS
             << std::endl;
        sout << "\tdsolve::SOLVER_CONSTRAINT_NUM_VARS: "
             << dsolve::SOLVER_CONSTRAINT_NUM_VARS << std::endl;
        sout << "\tdsolve::SOLVER_PROFILE_OUTPUT_FREQ: "
             << dsolve::SOLVER_PROFILE_OUTPUT_FREQ << std::endl;
        sout << "\tdsolve::SOLVER_RK45_STAGES: " << dsolve::SOLVER_RK45_STAGES
             << std::endl;
        sout << "\tdsolve::SOLVER_RK4_STAGES: " << dsolve::SOLVER_RK4_STAGES
             << std::endl;
        sout << "\tdsolve::SOLVER_RK3_STAGES: " << dsolve::SOLVER_RK3_STAGES
             << std::endl;
        sout << "\tdsolve::SOLVER_SAFETY_FAC: " << dsolve::SOLVER_SAFETY_FAC
             << std::endl;
        sout << "\tdsolve::SOLVER_NUM_VARS_INTENL: "
             << dsolve::SOLVER_NUM_VARS_INTENL << std::endl;
        sout << "\tdsolve::SOLVER_COMPD_MIN: [";
        for (unsigned int i = 0; i < 3; ++i) {
            sout << dsolve::SOLVER_COMPD_MIN[i] << (i < 3 - 1 ? ',' : ']');
        }
        sout << std::endl;
        sout << "\tdsolve::SOLVER_COMPD_MAX: [";
        for (unsigned int i = 0; i < 3; ++i) {
            sout << dsolve::SOLVER_COMPD_MAX[i] << (i < 3 - 1 ? ',' : ']');
        }
        sout << std::endl;
        sout << "\tdsolve::SOLVER_OCTREE_MIN: [";
        for (unsigned int i = 0; i < 3; ++i) {
            sout << dsolve::SOLVER_OCTREE_MIN[i] << (i < 3 - 1 ? ',' : ']');
        }
        sout << std::endl;
        sout << "\tdsolve::SOLVER_OCTREE_MAX: [";
        for (unsigned int i = 0; i < 3; ++i) {
            sout << dsolve::SOLVER_OCTREE_MAX[i] << (i < 3 - 1 ? ',' : ']');
        }
        sout << std::endl;
        sout << "\tdsolve::SOLVER_IO_OUTPUT_FREQ: "
             << dsolve::SOLVER_IO_OUTPUT_FREQ << std::endl;
        sout << "\tdsolve::SOLVER_TIME_STEP_OUTPUT_FREQ: "
             << dsolve::SOLVER_TIME_STEP_OUTPUT_FREQ << std::endl;

        sout << "\tdsolve::SOLVER_NUM_CONSOLE_OUTPUT_VARS: "
             << dsolve::SOLVER_NUM_CONSOLE_OUTPUT_VARS << std::endl;
        sout << "\tdsolve::SOLVER_CONSOLE_OUTPUT_VARS: [";
        for (unsigned int i = 0; i < dsolve::SOLVER_NUM_CONSOLE_OUTPUT_VARS;
             ++i) {
            sout << dsolve::SOLVER_CONSOLE_OUTPUT_VARS[i]
                 << (i < dsolve::SOLVER_NUM_CONSOLE_OUTPUT_VARS - 1 ? ','
                                                                    : ']');
        }
        sout << std::endl;
        sout << "\tdsolve::SOLVER_NUM_CONSOLE_OUTPUT_CONSTRAINTS: "
             << dsolve::SOLVER_NUM_CONSOLE_OUTPUT_CONSTRAINTS << std::endl;
        sout << "\tdsolve::SOLVER_CONSOLE_OUTPUT_CONSTRAINTS: [";
        for (unsigned int i = 0;
             i < dsolve::SOLVER_NUM_CONSOLE_OUTPUT_CONSTRAINTS; ++i) {
            sout << dsolve::SOLVER_CONSOLE_OUTPUT_CONSTRAINTS[i]
                 << (i < dsolve::SOLVER_NUM_CONSOLE_OUTPUT_CONSTRAINTS - 1
                         ? ','
                         : ']');
        }
        sout << std::endl;

        sout << "\tdsolve::SOLVER_REMESH_TEST_FREQ: "
             << dsolve::SOLVER_REMESH_TEST_FREQ << std::endl;
        sout << "\tdsolve::SOLVER_CHECKPT_FREQ: " << dsolve::SOLVER_CHECKPT_FREQ
             << std::endl;
        sout << "\tdsolve::SOLVER_RESTORE_SOLVER: "
             << dsolve::SOLVER_RESTORE_SOLVER << std::endl;
        sout << "\tdsolve::SOLVER_ENABLE_BLOCK_ADAPTIVITY: "
             << dsolve::SOLVER_ENABLE_BLOCK_ADAPTIVITY << std::endl;
        sout << "\tdsolve::SOLVER_VTU_FILE_PREFIX: "
             << dsolve::SOLVER_VTU_FILE_PREFIX << std::endl;
        sout << "\tdsolve::SOLVER_CHKPT_FILE_PREFIX: "
             << dsolve::SOLVER_CHKPT_FILE_PREFIX << std::endl;
        sout << "\tdsolve::SOLVER_PROFILE_FILE_PREFIX: "
             << dsolve::SOLVER_PROFILE_FILE_PREFIX << std::endl;
        sout << "\tdsolve::SOLVER_NUM_REFINE_VARS: "
             << dsolve::SOLVER_NUM_REFINE_VARS << std::endl;
        sout << "\tdsolve::SOLVER_REFINE_VARIABLE_INDICES: [";
        for (unsigned int i = 0; i < dsolve::SOLVER_NUM_REFINE_VARS; ++i) {
            sout << dsolve::SOLVER_REFINE_VARIABLE_INDICES[i]
                 << (i < dsolve::SOLVER_NUM_REFINE_VARS - 1 ? ',' : ']');
        }
        sout << std::endl;
        sout << "\tdsolve::SOLVER_NUM_EVOL_VARS_VTU_OUTPUT: "
             << dsolve::SOLVER_NUM_EVOL_VARS_VTU_OUTPUT << std::endl;
        sout << "\tdsolve::SOLVER_NUM_CONST_VARS_VTU_OUTPUT: "
             << dsolve::SOLVER_NUM_CONST_VARS_VTU_OUTPUT << std::endl;
        sout << "\tdsolve::SOLVER_VTU_OUTPUT_EVOL_INDICES: [";
        for (unsigned int i = 0; i < dsolve::SOLVER_NUM_EVOL_VARS_VTU_OUTPUT;
             ++i) {
            sout << dsolve::SOLVER_VTU_OUTPUT_EVOL_INDICES[i]
                 << (i < dsolve::SOLVER_NUM_EVOL_VARS_VTU_OUTPUT - 1 ? ','
                                                                     : ']');
        }
        sout << std::endl;
        sout << "\tdsolve::SOLVER_VTU_OUTPUT_CONST_INDICES: [";
        for (unsigned int i = 0; i < dsolve::SOLVER_NUM_CONST_VARS_VTU_OUTPUT;
             ++i) {
            sout << dsolve::SOLVER_VTU_OUTPUT_CONST_INDICES[i]
                 << (i < dsolve::SOLVER_NUM_CONST_VARS_VTU_OUTPUT - 1 ? ','
                                                                      : ']');
        }
        sout << std::endl;
        sout << "\tdsolve::SOLVER_IO_OUTPUT_GAP: "
             << dsolve::SOLVER_IO_OUTPUT_GAP << std::endl;
        sout << "\tdsolve::SOLVER_DENDRO_GRAIN_SZ: "
             << dsolve::SOLVER_DENDRO_GRAIN_SZ << std::endl;
        sout << "\tdsolve::SOLVER_DENDRO_AMR_FAC: "
             << dsolve::SOLVER_DENDRO_AMR_FAC << std::endl;
        sout << "\tdsolve::SOLVER_INIT_GRID_ITER: "
             << dsolve::SOLVER_INIT_GRID_ITER << std::endl;
        sout << "\tdsolve::SOLVER_INIT_GRID_REINITIALIZE_EACH_TIME: "
             << dsolve::SOLVER_INIT_GRID_REINITIALIZE_EACH_TIME << std::endl;
        sout << "\tdsolve::SOLVER_SPLIT_FIX: " << dsolve::SOLVER_SPLIT_FIX
             << std::endl;
        sout << "\tdsolve::SOLVER_CFL_FACTOR: " << dsolve::SOLVER_CFL_FACTOR
             << std::endl;
        sout << "\tdsolve::SOLVER_RK_TIME_BEGIN: "
             << dsolve::SOLVER_RK_TIME_BEGIN << std::endl;
        sout << "\tdsolve::SOLVER_RK_TIME_END: " << dsolve::SOLVER_RK_TIME_END
             << std::endl;
        sout << "\tdsolve::SOLVER_RK_TYPE: " << dsolve::SOLVER_RK_TYPE
             << std::endl;
        sout << "\tdsolve::SOLVER_RK45_TIME_STEP_SIZE: "
             << dsolve::SOLVER_RK45_TIME_STEP_SIZE << std::endl;
        sout << "\tdsolve::SOLVER_RK45_DESIRED_TOL: "
             << dsolve::SOLVER_RK45_DESIRED_TOL << std::endl;
        sout << "\tdsolve::DISSIPATION_TYPE: " << dsolve::DISSIPATION_TYPE
             << std::endl;
        sout << "\tdsolve::SOLVER_DISSIPATION_NC: "
             << dsolve::SOLVER_DISSIPATION_NC << std::endl;
        sout << "\tdsolve::SOLVER_DISSIPATION_S: "
             << dsolve::SOLVER_DISSIPATION_S << std::endl;
        sout << "\tdsolve::SOLVER_LTS_TS_OFFSET: "
             << dsolve::SOLVER_LTS_TS_OFFSET << std::endl;
        sout << "\tdsolve::SOLVER_VTU_Z_SLICE_ONLY: "
             << dsolve::SOLVER_VTU_Z_SLICE_ONLY << std::endl;
        sout << "\tdsolve::SOLVER_ASYNC_COMM_K: " << dsolve::SOLVER_ASYNC_COMM_K
             << std::endl;
        sout << "\tdsolve::SOLVER_LOAD_IMB_TOL: " << dsolve::SOLVER_LOAD_IMB_TOL
             << std::endl;
        sout << "\tdsolve::SOLVER_DIM: " << dsolve::SOLVER_DIM << std::endl;
        sout << "\tdsolve::SOLVER_MAXDEPTH: " << dsolve::SOLVER_MAXDEPTH
             << std::endl;
        sout << "\tdsolve::SOLVER_MINDEPTH: " << dsolve::SOLVER_MINDEPTH
             << std::endl;
        sout << "\tdsolve::SOLVER_WAVELET_TOL: " << dsolve::SOLVER_WAVELET_TOL
             << std::endl;
        sout << "\tdsolve::SOLVER_USE_WAVELET_TOL_FUNCTION: "
             << dsolve::SOLVER_USE_WAVELET_TOL_FUNCTION << std::endl;
        sout << "\tdsolve::SOLVER_WAVELET_TOL_MAX: "
             << dsolve::SOLVER_WAVELET_TOL_MAX << std::endl;
        sout << "\tdsolve::SOLVER_WAVELET_TOL_FUNCTION_R0: "
             << dsolve::SOLVER_WAVELET_TOL_FUNCTION_R0 << std::endl;
        sout << "\tdsolve::SOLVER_WAVELET_TOL_FUNCTION_R1: "
             << dsolve::SOLVER_WAVELET_TOL_FUNCTION_R1 << std::endl;
        sout << "\tdsolve::SOLVER_USE_FD_GRID_TRANSFER: "
             << dsolve::SOLVER_USE_FD_GRID_TRANSFER << std::endl;
        sout << "\tdsolve::SOLVER_ETA_CONST: " << dsolve::SOLVER_ETA_CONST
             << std::endl;
        sout << "\tdsolve::SOLVER_ETA_R0: " << dsolve::SOLVER_ETA_R0
             << std::endl;
        sout << "\tdsolve::SOLVER_ETA_DAMPING_EXP: "
             << dsolve::SOLVER_ETA_DAMPING_EXP << std::endl;
        sout << "\tdsolve::SOLVER_REFINEMENT_MODE: "
             << dsolve::SOLVER_REFINEMENT_MODE << std::endl;
        sout << "\tdsolve::SOLVER_BLK_MIN_X: " << dsolve::SOLVER_BLK_MIN_X
             << std::endl;
        sout << "\tdsolve::SOLVER_BLK_MIN_Y: " << dsolve::SOLVER_BLK_MIN_Y
             << std::endl;
        sout << "\tdsolve::SOLVER_BLK_MIN_Z: " << dsolve::SOLVER_BLK_MIN_Z
             << std::endl;
        sout << "\tdsolve::SOLVER_BLK_MAX_X: " << dsolve::SOLVER_BLK_MAX_X
             << std::endl;
        sout << "\tdsolve::SOLVER_BLK_MAX_Y: " << dsolve::SOLVER_BLK_MAX_Y
             << std::endl;
        sout << "\tdsolve::SOLVER_BLK_MAX_Z: " << dsolve::SOLVER_BLK_MAX_Z
             << std::endl;
        sout << "\tdsolve::KO_DISS_SIGMA: " << dsolve::KO_DISS_SIGMA
             << std::endl;
        sout << "\tdsolve::SOLVER_ID_TYPE: " << dsolve::SOLVER_ID_TYPE
             << std::endl;
        sout << "\tdsolve::SOLVER_GRID_MIN_X: " << dsolve::SOLVER_GRID_MIN_X
             << std::endl;
        sout << "\tdsolve::SOLVER_GRID_MAX_X: " << dsolve::SOLVER_GRID_MAX_X
             << std::endl;
        sout << "\tdsolve::SOLVER_GRID_MIN_Y: " << dsolve::SOLVER_GRID_MIN_Y
             << std::endl;
        sout << "\tdsolve::SOLVER_GRID_MAX_Y: " << dsolve::SOLVER_GRID_MAX_Y
             << std::endl;
        sout << "\tdsolve::SOLVER_GRID_MIN_Z: " << dsolve::SOLVER_GRID_MIN_Z
             << std::endl;
        sout << "\tdsolve::SOLVER_GRID_MAX_Z: " << dsolve::SOLVER_GRID_MAX_Z
             << std::endl;
        sout << "\tDERIVS USE: " << SOLVER_DERIVS->toString() << std::endl;
    }
}
}  // namespace dsolve

//[[[end]]]

// end parameters.cpp file
