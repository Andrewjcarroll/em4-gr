/**
 * @file em4_params_cu.h
 * @brief Contains GPU related parameters and coordinate macros for EM4.
 */

#pragma once

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

#include <iostream>
#include "cuda_runtime.h"
#include "point.h"

// Forward declare the global variables (these should match parameters.h/cpp)
namespace cuda {
namespace _em4 {
extern __constant__ double __SOLVER_COMPD_MIN[3];
extern __constant__ double __SOLVER_COMPD_MAX[3];
extern __constant__ double __SOLVER_OCTREE_MIN[3];
extern __constant__ double __SOLVER_OCTREE_MAX[3];
}  // namespace _em4
}  // namespace cuda

// -------------------------------------------------------------------------
// Coordinate Mapping Macros (Aligned with grDef.h)
// -------------------------------------------------------------------------

#define __Rx (cuda::_em4::__SOLVER_COMPD_MAX[0] - cuda::_em4::__SOLVER_COMPD_MIN[0])
#define __Ry (cuda::_em4::__SOLVER_COMPD_MAX[1] - cuda::_em4::__SOLVER_COMPD_MIN[1])
#define __Rz (cuda::_em4::__SOLVER_COMPD_MAX[2] - cuda::_em4::__SOLVER_COMPD_MIN[2])

#define __RgX (cuda::_em4::__SOLVER_OCTREE_MAX[0] - cuda::_em4::__SOLVER_OCTREE_MIN[0])
#define __RgY (cuda::_em4::__SOLVER_OCTREE_MAX[1] - cuda::_em4::__SOLVER_OCTREE_MIN[1])
#define __RgZ (cuda::_em4::__SOLVER_OCTREE_MAX[2] - cuda::_em4::__SOLVER_OCTREE_MIN[2])

#define __GRIDX_TO_X(xg)                                           \
    (((__Rx / __RgX) * (xg - cuda::_em4::__SOLVER_OCTREE_MIN[0])) + \
     cuda::_em4::__SOLVER_COMPD_MIN[0])
#define __GRIDY_TO_Y(yg)                                           \
    (((__Ry / __RgY) * (yg - cuda::_em4::__SOLVER_OCTREE_MIN[1])) + \
     cuda::_em4::__SOLVER_COMPD_MIN[1])
#define __GRIDZ_TO_Z(zg)                                           \
    (((__Rz / __RgZ) * (zg - cuda::_em4::__SOLVER_OCTREE_MIN[2])) + \
     cuda::_em4::__SOLVER_COMPD_MIN[2])

#define __X_TO_GRIDX(xc)                                          \
    (((__RgX / __Rx) * (xc - cuda::_em4::__SOLVER_COMPD_MIN[0])) + \
     cuda::_em4::__SOLVER_OCTREE_MIN[0])
#define __Y_TO_GRIDY(yc)                                          \
    (((__RgY / __Ry) * (yc - cuda::_em4::__SOLVER_COMPD_MIN[1])) + \
     cuda::_em4::__SOLVER_OCTREE_MIN[1])
#define __Z_TO_GRIDZ(zc)                                          \
    (((__RgZ / __Rz) * (zc - cuda::_em4::__SOLVER_COMPD_MIN[2])) + \
     cuda::_em4::__SOLVER_OCTREE_MIN[2])

namespace cuda {

/**
 * @brief Compute parameters needed for EM4 equation evaluation on the GPU.
 */
struct SOLVERComputeParams {
    double kappa_1;
    double kappa_2;
    double KO_DISS_SIGMA;
};

// -------------------------------------------------------------------------
// Device Global Pointers (Matching nlsm pattern)
// -------------------------------------------------------------------------

/**stores the device properties*/
extern cudaDeviceProp* __CUDA_DEVICE_PROPERTIES;

/** number of evol vars */
extern unsigned int* __SOLVER_NUM_VARS;

/**compute parameters*/
extern SOLVERComputeParams* __SOLVER_COMPUTE_PARMS;

}  // end of namespace cuda
