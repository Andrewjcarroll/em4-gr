/**
 * @file gpu_profiler.cuh
 * @brief Lightweight CUDA event-based timer for profiling GPU kernel time.
 *
 * Wraps a cudaEvent_t start/stop pair and accumulates elapsed time across
 * multiple calls, mirroring the CPU profiler_t interface from dendrolib.
 *
 * Usage:
 *   gpu_profiler_t t;
 *   t.create();
 *   t.start();
 *   kernel<<<...>>>(...);
 *   t.stop();
 *   t.sync();               // blocks until event is done, accumulates time
 *   printf("%f ms\n", t.ms);
 *   t.destroy();
 */

#pragma once

#ifdef __CUDACC__

#include <cuda_runtime.h>
#include <stdio.h>

#define GPU_PROF_CHECK(call)                                          \
    do {                                                              \
        cudaError_t _e = (call);                                      \
        if (_e != cudaSuccess) {                                      \
            printf("[gpu_profiler] CUDA error %s:%d: %s\n", __FILE__, \
                   __LINE__, cudaGetErrorString(_e));                 \
        }                                                             \
    } while (0)

namespace dsolve {

struct gpu_profiler_t {
    cudaEvent_t ev_start;
    cudaEvent_t ev_stop;
    float ms;
    long long num_calls;

    bool _created;
    bool _started;

    gpu_profiler_t()
        : ms(0.0f), num_calls(0), _created(false), _started(false) {}

    void create() {
        if (!_created) {
            GPU_PROF_CHECK(cudaEventCreate(&ev_start));
            GPU_PROF_CHECK(cudaEventCreate(&ev_stop));
            _created = true;
        }
    }

    void destroy() {
        if (_created) {
            GPU_PROF_CHECK(cudaEventDestroy(ev_start));
            GPU_PROF_CHECK(cudaEventDestroy(ev_stop));
            _created = false;
            _started = false;
        }
    }

    void start() {
        if (_created) {
            GPU_PROF_CHECK(cudaEventRecord(ev_start, 0));
            _started = true;
        }
    }

    void stop() {
        if (_created && _started) {
            GPU_PROF_CHECK(cudaEventRecord(ev_stop, 0));
        }
    }

    void sync() {
        if (_created && _started) {
            GPU_PROF_CHECK(cudaEventSynchronize(ev_stop));
            float elapsed = 0.0f;
            GPU_PROF_CHECK(cudaEventElapsedTime(&elapsed, ev_start, ev_stop));
            ms += elapsed;
            num_calls++;
            _started = false;
        }
    }

    void clear() {
        ms        = 0.0f;
        num_calls = 0;
        _started  = false;
    }

    double seconds() const { return static_cast<double>(ms) / 1000.0; }
};

}  // namespace dsolve

#endif  // __CUDACC__
