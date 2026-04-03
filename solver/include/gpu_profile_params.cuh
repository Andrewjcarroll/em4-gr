/**
 * @file gpu_profile_params.cuh
 * @brief GPU event-timer instances for EM4 GPU kernel profiling.
 *
 * These timers measure time spent purely on the GPU (kernel execution,
 * memory transfers). They complement the CPU-side profiler_t instances
 * in profile_params.h.
 *
 * All instances are defined in EM4CtxGPU.cu and accumulated per-rank.
 * For multi-rank reporting, use MPI_Reduce/Allreduce on the ms field.
 */

#pragma once

#ifdef __CUDACC__
#include "gpu_profiler.cuh"
#endif

namespace ot {
class Mesh;
}

namespace dsolve {
namespace gpu_timer {

#ifdef __CUDACC__

extern gpu_profiler_t t_h2d;
extern gpu_profiler_t t_d2h;

extern gpu_profiler_t t_unzip;
extern gpu_profiler_t t_deriv_x;
extern gpu_profiler_t t_deriv_y;
extern gpu_profiler_t t_deriv_z;
extern gpu_profiler_t t_rhs_kernel;
extern gpu_profiler_t t_zip;

#endif  // __CUDACC__

void initGPUTimers();
void destroyGPUTimers();
void resetGPUTimers();
void profileInfoGPU(const char* filePrefix, const ot::Mesh* pMesh);

}  // namespace gpu_timer
}  // namespace dsolve
