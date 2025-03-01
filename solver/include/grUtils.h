/**
 * @author Milinda Fernando / David Van Komen
 * School of Computing, University of Utah
 * @brief Contins utility functions for simulations
 */

#ifndef SFCSORTBENCH_GRUTILS_H
#define SFCSORTBENCH_GRUTILS_H

#include "block.h"
#include "dendroProfileParams.h"
#include "grDef.h"
#include "json.hpp"
#include "lebedev.h"
#include "mesh.h"
#include "meshUtils.h"
#include "parUtils.h"
#include "parameters.h"
#include "point.h"
#include "profile_params.h"
#include "swsh.h"

#define RAISE_ERROR(msg)                                                       \
    std::cout << "[Error]: " << __FILE__ << ":" << __LINE__ << " at function " \
              << __FUNCTION__ << " (" << msg << ")" << std::endl

using json = nlohmann::json;
namespace dsolve {
/**
 * @brief: Read the parameter file and initialize the variables in parameters.h
 * file.
 * @param[in] fName: file name
 * @param[in] comm: MPI communicator.
 * */
// void readParamFile(const char *fName, MPI_Comm comm);

/**
 * @brief dump the read parameter files.
 *
 * @param sout
 * @param root
 * @param comm
 */
// void dumpParamFile(std::ostream &sout, int root, MPI_Comm comm);

// clang-format off
/*[[[cog
import cog
import sys
import os
import importlib.util
import dendrosym

# get the current working directory, should be root of project
current_path = os.getcwd()
output_path = os.path.join(current_path, "gencode")

# the following lines will import any module directly from
spec = importlib.util.spec_from_file_location("dendroconf", CONFIG_FILE_PATH)
dendroconf = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = dendroconf
spec.loader.exec_module(dendroconf)

cog.outl("// INITIAL DATA FUNCTIONS")
cog.outl(dendroconf.dendroConfigs.generate_initial_data_declaration(var_type="evolution"))
]]]*/

//[[[end]]]


/**
 * @brief Wrapper function for the analytical solve for EM4's initial data
 */
void analyticalSolEM4_BLOCK(double **uZipAnalyticVars,
                            double **uZipAnalyticDiffVars,
                            const double **uZipVars, const double time, const unsigned int &offset,
                            const double *pmin, const double *pmax,
                            const unsigned int *sz, const unsigned int &bflag);


/**
 * @brief Computes the analytical solution for the E and B fields at a given point and time.
 *
 * This function calculates the analytical solution of the electromagnetic field variables at a specified spatial
 * point defined in *dendro grid format* (xx, yy, zz) and time (t) and stores the results in the array 'var'.
 *
 * @param[in] xx X-coordinate in GRIDX form.
 * @param[in] yy Y-coordinate in GRIDY form. 
 * @param[in] zz Z-coordinate in GRIDZ form.
 * @param[in] t Time at which the solution is evaluated.
 * @param[out] var Array to store the computed electromagnetic field variables.
 * @param[in] varsAreGrid If the variables are in "grid" form and not true form (default true).
 */
void analyticalSolEM4(const double xx, const double yy, const double zz,
                      const double t, double *var, bool varsAreGrid = true);

/**
 * @brief Initializes E and B fields for EM4 at a specified spatial point.
 *
 * This function initializes electromagnetic field variables at the given,
 * spatial point (x, y, z) by storing the results in the array 'var'.
 *
 * @param[in] x X-coordinate of the spatial point.
 * @param[in] y Y-coordinate of the spatial point.
 * @param[in] z Z-coordinate of the spatial point.
 * @param[out] var Array to store the field variables.
 */
void initDataEM4(const double x, const double y, const double z,
              double* var);

/**
 * @brief calculate and set the initial data for superposed boosted kerr-sen
 * @param xx1 : x coord, GRIDX format
 * @param yy1 : y coord, GRIDX format
 * @param zz1 : z coord, GRIDX format
 * @param var : initialized solver variables for the grid points
 */
void initDataFuncToPhysCoords(double xx1, double yy1, double zz1, double *var);

/**
 * @brief: Generates block adaptive octree for the given binary blockhole
 * problem.
 * @param[out] tmpNodes: created octree tmpNodes
 * @param[in] pt_min: block min point
 * @param[in] pt_max: block max point
 * @param[in] regLev: regular grid level
 * @param[in] maxDepth: maximum refinement level.
 * @param[in] comm: MPI communicator.
 * */
void blockAdaptiveOctree(std::vector<ot::TreeNode> &tmpNodes,
                         const Point &pt_min, const Point &pt_max,
                         const unsigned int regLev, const unsigned int maxDepth,
                         MPI_Comm comm);

/**
 * @brief Compute the wavelet tolerance as a function of space.
 *
 * @param x : x coord.
 * @param y : y coord
 * @param z : z coord
 * @param tol_min : min. tolerance value.
 * @return double
 */
double computeWTol(double x, double y, double z, double tol_min);

/**
 * @brief Compute the wavelet tolerance as a function of space (uses actual
 * domain coordinates not octree coordinates)
 *
 * @param x : x coord.
 * @param y : y coord
 * @param z : z coord
 * @param hx : resolution in x,y,z
 * @return double
 */
double computeWTolDCoords(double x, double y, double z, double *hx);

/**
 * @breif: Compute L2 constraint norms.
 */
template <typename T>
double computeConstraintL2Norm(const T *constraintVec, const T *maskVec,
                               unsigned int lbegin, unsigned int lend,
                               MPI_Comm comm);

/**
 * @brief Compute L2 constraint norms.
 */
template <typename T>
double computeConstraintL2Norm(const ot::Mesh *mesh, const T *constraintVec,
                               const T *maskVector, T maskthreshoold);

/**
 * @breif write constraints to a file.
 */
template <typename T>
double extractConstraints(const ot::Mesh *mesh, const T **constraintVar,
                          const T *maskVec, double maskthreshoold,
                          unsigned int timestep, double stime);

/**@brief : write a block to binary*/
void writeBLockToBinary(const double **unzipVarsRHS, unsigned int offset,
                        const double *pmin, const double *pmax, double *bxMin,
                        double *bxMax, const unsigned int *sz,
                        unsigned int blkSz, double dxFactor,
                        const char *fprefix);

/**@brief returns the octant weight for LTS timestepping. */
unsigned int getOctantWeight(const ot::TreeNode *pNode);


/**
 * @brief Allocate the derivative workspace for use in RHS functionality
 * 
 * @param pMesh 
 * @param s_fac 
 */
void allocate_deriv_workspace(const ot::Mesh *pMesh, unsigned int s_fac);

/**
 * @brief Deallocate the derivative workspace for use in the RHS functionality
 * 
 */
void deallocate_deriv_workspace();

ot::Mesh *weakScalingReMesh(ot::Mesh *pMesh, unsigned int target_npes);

}  // end of namespace dsolve

namespace dsolve {

namespace timer {

/**@brief initialize all the flop counters. */
void initFlops();

/**@brief clears the snapshot counter for time profiler variables*/
void resetSnapshot();

/**@brief reduce min mean max.
 * @param [in] stat: local time
 * @param [out] stat_g 0-min, 1-mean 2-max
 * */
template <typename T>
void computeOverallStats(T *stat, T *stat_g, MPI_Comm comm) {
    int rank, npes;
    MPI_Comm_size(comm, &npes);
    MPI_Comm_rank(comm, &rank);

    par::Mpi_Reduce(stat, stat_g, 1, MPI_MIN, 0, comm);
    par::Mpi_Reduce(stat, stat_g + 1, 1, MPI_SUM, 0, comm);
    par::Mpi_Reduce(stat, stat_g + 2, 1, MPI_MAX, 0, comm);
    stat_g[1] /= (npes);
}

/** @breif : printout the profile parameters. */
void profileInfo(const char *filePrefix, const ot::Mesh *pMesh);

/** @breif : printout the profile parameters (intermediate profile information).
 */
void profileInfoIntermediate(const char *filePrefix, const ot::Mesh *pMesh,
                             const unsigned int currentStep);

}  // namespace timer

}  // namespace dsolve


#include "grUtils.tcc"

#endif  // SFCSORTBENCH_GRUTILS_H
