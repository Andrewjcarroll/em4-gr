/**
 * @file EM4CtxGPU.cuh
 * @brief EM4 rhs context file for CUDA.
 */

#pragma once

#include <iostream>
#include <vector>

#include "bc_cu.cuh"
#include "checkPoint.h"
#include "ctx.h"
#include "derivs_cu.cuh"
#include "device.h"
#include "em4_params_cu.h"
#include "grUtils.h"
#include "mesh.h"
#include "mesh_gpu.cuh"
#include "parameters.h"

namespace cuda {

/**
 * @brief EVAR_DERIVS struct to store pointers for derivatives on the GPU.
 */
struct EVAR_DERIVS {
    // Derivatives for E0, E1, E2
    DEVICE_REAL *grad_0_E0, *grad_1_E0, *grad_2_E0;
    DEVICE_REAL *grad_0_E1, *grad_1_E1, *grad_2_E1;
    DEVICE_REAL *grad_0_E2, *grad_1_E2, *grad_2_E2;

    // Derivatives for B0, B1, B2
    DEVICE_REAL *grad_0_B0, *grad_1_B0, *grad_2_B0;
    DEVICE_REAL *grad_0_B1, *grad_1_B1, *grad_2_B1;
    DEVICE_REAL *grad_0_B2, *grad_1_B2, *grad_2_B2;

    // Derivatives for Phi, Psi
    DEVICE_REAL *grad_0_Phi, *grad_1_Phi, *grad_2_Phi;
    DEVICE_REAL *grad_0_Psi, *grad_1_Psi, *grad_2_Psi;

    // KO Dissipation Derivatives for E0, E1, E2
    DEVICE_REAL *ko_grad_0_E0, *ko_grad_1_E0, *ko_grad_2_E0;
    DEVICE_REAL *ko_grad_0_E1, *ko_grad_1_E1, *ko_grad_2_E1;
    DEVICE_REAL *ko_grad_0_E2, *ko_grad_1_E2, *ko_grad_2_E2;

    // KO Dissipation Derivatives for B0, B1, B2
    DEVICE_REAL *ko_grad_0_B0, *ko_grad_1_B0, *ko_grad_2_B0;
    DEVICE_REAL *ko_grad_0_B1, *ko_grad_1_B1, *ko_grad_2_B1;
    DEVICE_REAL *ko_grad_0_B2, *ko_grad_1_B2, *ko_grad_2_B2;

    // KO Dissipation Derivatives for Phi, Psi
    DEVICE_REAL *ko_grad_0_Phi, *ko_grad_1_Phi, *ko_grad_2_Phi;
    DEVICE_REAL *ko_grad_0_Psi, *ko_grad_1_Psi, *ko_grad_2_Psi;
};

/**
 * @brief VL enum for EM4-GR GPU variables.
 */
enum VL { CPU_EV = 0, CPU_EV_UZ, GPU_EV, GPU_EV_UZ_IN, GPU_EV_UZ_OUT, END };

typedef ot::DVector<DendroScalar, unsigned int> DVec;

class EM4CtxGPU : public ts::Ctx<EM4CtxGPU, DendroScalar, unsigned int> {
   protected:
    DendroIntL m_uiGlobalMeshElements = 0;
    DendroIntL m_uiGlobalGridPoints   = 0;
    bool m_uiWroteGridInfoHeader      = false;

   protected:
    device::MeshGPU m_mesh_cpu;

    /**@brief: mesh in the device*/
    device::MeshGPU* m_dptr_mesh;

    /**@brief: evolution var (zip)*/
    DVec m_var[VL::END];

    static const unsigned int DEVICE_RHS_BATCHED_GRAIN_SZ = 512;
    static const unsigned int DEVICE_RHS_BLK_SZ           = 13 * 13 * 13;
    static const unsigned int DEVICE_RHS_NSTREAMS         = 1;

    EVAR_DERIVS* m_deriv_evars                            = nullptr;
    EVAR_DERIVS* m_dptr_deriv_evars                       = nullptr;
    DEVICE_REAL* m_dptr_deriv_base                        = nullptr;

   public:
    /**@brief: default constructor*/
    EM4CtxGPU(ot::Mesh* pMesh);

    /**@brief: default deconstructor*/
    ~EM4CtxGPU();

    /**@brief: initial solution*/
    int initialize();

    /**@brief: initialize the grid, solution. */
    int init_grid();

    /** @brief: Any flags that need to be adjusted and updated for the next step
     */
    void resetForNextStep() { dsolve::timer::resetSnapshot(); }

    void resetForEvolutionStuff() {}

    /**
     * @brief computes the rhs
     *
     * @param in : zipped input
     * @param out : zipped output
     * @param sz  : number of variables.
     * @param time : current time.
     * @return int : status. (0) on success.
     */
    int rhs(DVec* in, DVec* out, unsigned int sz, DendroScalar time);

    /**@brief: function execute before each stage
     * @param sIn: stage var in.
     */
    inline int pre_stage(DVec& sIn) { return 0; }

    /**@brief: function execute after each stage
     * @param sIn: stage var in.
     */
    int post_stage(DVec& sIn);

    /**@brief: function execute before each step*/
    inline int pre_timestep(DVec& sIn) { return 0; }

    /**@brief: function execute after each step*/
    int post_timestep(DVec& sIn);

    /**@brief: function execute after each step*/
    bool is_remesh();

    /**@brief: write to vtu. */
    int write_vtu();

    /**@brief: writes checkpoint*/
    int write_checkpt();

    /**@brief: restore from check point*/
    int restore_checkpt();

    /**@brief: should be called for free up the contex memory. */
    int finalize();

    /**@brief: pack and returns the evolution variables to one DVector*/
    DVec& get_evolution_vars();

    /**@brief: pack and returns the CPU evolution variables to one DVector*/
    DVec& get_evolution_vars_cpu();

    /**@brief: pack and returns the constraint variables to one DVector*/
    DVec& get_constraint_vars();

    /**@brief: pack and returns the primitive variables to one DVector*/
    DVec& get_primitive_vars();

    /**@brief: prints any messages to the terminal output. */
    int terminal_output();

    /**@brief: returns the async communication batch size. */
    unsigned int get_async_batch_sz() { return dsolve::SOLVER_ASYNC_COMM_K; }

    /**@brief: returns the number of variables considered when performing
     * refinement*/
    unsigned int get_num_refine_vars() {
        return dsolve::SOLVER_NUM_REFINE_VARS;
    }

    /**@brief: return the pointer for containing evolution refinement variable
     * ids*/
    const unsigned int* get_refine_var_ids() {
        return dsolve::SOLVER_REFINE_VARIABLE_INDICES;
    }

    /**@brief return the wavelet tolerance function / value*/
    std::function<double(double, double, double, double*)> get_wtol_function() {
        double wtol = dsolve::SOLVER_WAVELET_TOL;
        std::function<double(double, double, double, double*)> waveletTolFunc =
            [wtol](double x, double y, double z, double* hx) {
                return 0.0;  // dsolve::computeWTol(x, y, z, hx); // TODO
            };
        return waveletTolFunc;
    }

    static unsigned int getBlkTimestepFac(unsigned int blev, unsigned int lmin,
                                          unsigned int lmax);

    int grid_transfer(const ot::Mesh* m_new);

    void calculate_full_grid_size();

    void write_grid_summary_data();

    int host_to_device_sync();
    int device_to_host_sync();

    int host_to_device_async(cudaStream_t s);
    int device_to_host_async(cudaStream_t s);

    inline device::MeshGPU*& get_meshgpu_device_ptr() { return m_dptr_mesh; }

    inline device::MeshGPU* get_meshgpu_host_handle() { return &m_mesh_cpu; }
};
}  // namespace cuda
