//
// Created by David
//
/**
 * @author Milinda Fernando / David Van Komen
 * School of Computing, University of Utah
 * @brief Contins parameters for the solver timing
 */

#ifndef SFCSORTBENCH_PROFILE_PARAMS_H
#define SFCSORTBENCH_PROFILE_PARAMS_H

#include "profiler.h"

namespace dsolve {
namespace timer {
extern profiler_t total_runtime;

extern profiler_t t_f2o;
extern profiler_t t_cons;
extern profiler_t t_bal;
extern profiler_t t_mesh;

extern profiler_t t_rkSolve;
extern profiler_t t_ghostEx_sync;

extern profiler_t t_unzip_sync;
extern profiler_t t_unzip_async;

extern profiler_t t_deriv;
extern profiler_t t_rhs;

extern profiler_t t_bdyc;

extern profiler_t t_zip;
extern profiler_t t_rkStep;

extern profiler_t t_isReMesh;
extern profiler_t t_gridTransfer;
extern profiler_t t_ioVtu;
extern profiler_t t_ioCheckPoint;

}  // namespace timer
}  // namespace dsolve

#endif  // SFCSORTBENCH_PROFILE_PARAMS_H
