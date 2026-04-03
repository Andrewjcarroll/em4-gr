#include <cmath>

#include "EM4CtxGPU.cuh"
#include "device_utils.cuh"
#include "em4_params_cu.h"
#include "grUtils.h"
#include "meshUtils.h"
#include "oct2vtk.h"

namespace cuda {

// GPU Constants (Definitions for symbols declared extern in em4_params_cu.h)
namespace _em4 {
__constant__ double __SOLVER_COMPD_MIN[3];
__constant__ double __SOLVER_COMPD_MAX[3];
__constant__ double __SOLVER_OCTREE_MIN[3];
__constant__ double __SOLVER_OCTREE_MAX[3];
}  // namespace _em4

// Global pointers
cudaDeviceProp* __CUDA_DEVICE_PROPERTIES    = nullptr;
unsigned int* __SOLVER_NUM_VARS             = nullptr;
SOLVERComputeParams* __SOLVER_COMPUTE_PARMS = nullptr;

// Kernel template functions
template <int pw, int pencils, int pencil_sz, int BATCHED_BLOCKS_SZ>
GLOBAL_FUNC void launch_dir_x_deriv_kernel(
    const device::MeshGPU* const dptr_mesh, const DEVICE_REAL* const u,
    EVAR_DERIVS* deriv_evars, BlockGPU3D* blk, DEVICE_UINT blk_begin) {
    const DEVICE_UINT blk_k  = GPUDevice::block_id_z();
    const DEVICE_UINT BLK_ID = blk_begin + blk_k;
    const DEVICE_UINT BLK_SZ = (blk[BLK_ID].m_aligned_sz[0]) *
                               (blk[BLK_ID].m_aligned_sz[1]) *
                               (blk[BLK_ID].m_aligned_sz[2]);
    const DEVICE_UINT offset     = blk[BLK_ID].m_offset;
    const DEVICE_UINT sz_per_dof = dptr_mesh->m_oct_unzip_sz;

    const DEVICE_REAL* const E0  = &u[dsolve::VAR::U_E0 * sz_per_dof + offset];
    const DEVICE_REAL* const E1  = &u[dsolve::VAR::U_E1 * sz_per_dof + offset];
    const DEVICE_REAL* const E2  = &u[dsolve::VAR::U_E2 * sz_per_dof + offset];
    const DEVICE_REAL* const B0  = &u[dsolve::VAR::U_B0 * sz_per_dof + offset];
    const DEVICE_REAL* const B1  = &u[dsolve::VAR::U_B1 * sz_per_dof + offset];
    const DEVICE_REAL* const B2  = &u[dsolve::VAR::U_B2 * sz_per_dof + offset];
    const DEVICE_REAL* const Phi = &u[dsolve::VAR::U_PHI * sz_per_dof + offset];
    const DEVICE_REAL* const Psi = &u[dsolve::VAR::U_PSI * sz_per_dof + offset];

    device::__deriv644_x<pw, pencils, pencil_sz>(
        deriv_evars->grad_0_E0 + blk_k * BLK_SZ, E0, blk + BLK_ID);
    device::__deriv644_x<pw, pencils, pencil_sz>(
        deriv_evars->grad_0_E1 + blk_k * BLK_SZ, E1, blk + BLK_ID);
    device::__deriv644_x<pw, pencils, pencil_sz>(
        deriv_evars->grad_0_E2 + blk_k * BLK_SZ, E2, blk + BLK_ID);
    device::__deriv644_x<pw, pencils, pencil_sz>(
        deriv_evars->grad_0_B0 + blk_k * BLK_SZ, B0, blk + BLK_ID);
    device::__deriv644_x<pw, pencils, pencil_sz>(
        deriv_evars->grad_0_B1 + blk_k * BLK_SZ, B1, blk + BLK_ID);
    device::__deriv644_x<pw, pencils, pencil_sz>(
        deriv_evars->grad_0_B2 + blk_k * BLK_SZ, B2, blk + BLK_ID);
    device::__deriv644_x<pw, pencils, pencil_sz>(
        deriv_evars->grad_0_Phi + blk_k * BLK_SZ, Phi, blk + BLK_ID);
    device::__deriv644_x<pw, pencils, pencil_sz>(
        deriv_evars->grad_0_Psi + blk_k * BLK_SZ, Psi, blk + BLK_ID);

    // KO Dissipation
    device::__ko_deriv42_x<pw, pencils, pencil_sz>(
        deriv_evars->ko_grad_0_E0 + blk_k * BLK_SZ, E0, blk + BLK_ID);
    device::__ko_deriv42_x<pw, pencils, pencil_sz>(
        deriv_evars->ko_grad_0_E1 + blk_k * BLK_SZ, E1, blk + BLK_ID);
    device::__ko_deriv42_x<pw, pencils, pencil_sz>(
        deriv_evars->ko_grad_0_E2 + blk_k * BLK_SZ, E2, blk + BLK_ID);
    device::__ko_deriv42_x<pw, pencils, pencil_sz>(
        deriv_evars->ko_grad_0_B0 + blk_k * BLK_SZ, B0, blk + BLK_ID);
    device::__ko_deriv42_x<pw, pencils, pencil_sz>(
        deriv_evars->ko_grad_0_B1 + blk_k * BLK_SZ, B1, blk + BLK_ID);
    device::__ko_deriv42_x<pw, pencils, pencil_sz>(
        deriv_evars->ko_grad_0_B2 + blk_k * BLK_SZ, B2, blk + BLK_ID);
    device::__ko_deriv42_x<pw, pencils, pencil_sz>(
        deriv_evars->ko_grad_0_Phi + blk_k * BLK_SZ, Phi, blk + BLK_ID);
    device::__ko_deriv42_x<pw, pencils, pencil_sz>(
        deriv_evars->ko_grad_0_Psi + blk_k * BLK_SZ, Psi, blk + BLK_ID);
}

template <int pw, int pencils, int pencil_sz, int BATCHED_BLOCKS_SZ>
GLOBAL_FUNC void launch_dir_y_deriv_kernel(
    const device::MeshGPU* const dptr_mesh, const DEVICE_REAL* const u,
    EVAR_DERIVS* deriv_evars, BlockGPU3D* blk, DEVICE_UINT blk_begin) {
    const DEVICE_UINT blk_k  = GPUDevice::block_id_z();
    const DEVICE_UINT BLK_ID = blk_begin + blk_k;
    const DEVICE_UINT BLK_SZ = (blk[BLK_ID].m_aligned_sz[0]) *
                               (blk[BLK_ID].m_aligned_sz[1]) *
                               (blk[BLK_ID].m_aligned_sz[2]);
    const DEVICE_UINT offset     = blk[BLK_ID].m_offset;
    const DEVICE_UINT sz_per_dof = dptr_mesh->m_oct_unzip_sz;

    const DEVICE_REAL* const E0  = &u[dsolve::VAR::U_E0 * sz_per_dof + offset];
    const DEVICE_REAL* const E1  = &u[dsolve::VAR::U_E1 * sz_per_dof + offset];
    const DEVICE_REAL* const E2  = &u[dsolve::VAR::U_E2 * sz_per_dof + offset];
    const DEVICE_REAL* const B0  = &u[dsolve::VAR::U_B0 * sz_per_dof + offset];
    const DEVICE_REAL* const B1  = &u[dsolve::VAR::U_B1 * sz_per_dof + offset];
    const DEVICE_REAL* const B2  = &u[dsolve::VAR::U_B2 * sz_per_dof + offset];
    const DEVICE_REAL* const Phi = &u[dsolve::VAR::U_PHI * sz_per_dof + offset];
    const DEVICE_REAL* const Psi = &u[dsolve::VAR::U_PSI * sz_per_dof + offset];

    device::__deriv644_y<pw, pencils, pencil_sz>(
        deriv_evars->grad_1_E0 + blk_k * BLK_SZ, E0, blk + BLK_ID);
    device::__deriv644_y<pw, pencils, pencil_sz>(
        deriv_evars->grad_1_E1 + blk_k * BLK_SZ, E1, blk + BLK_ID);
    device::__deriv644_y<pw, pencils, pencil_sz>(
        deriv_evars->grad_1_E2 + blk_k * BLK_SZ, E2, blk + BLK_ID);
    device::__deriv644_y<pw, pencils, pencil_sz>(
        deriv_evars->grad_1_B0 + blk_k * BLK_SZ, B0, blk + BLK_ID);
    device::__deriv644_y<pw, pencils, pencil_sz>(
        deriv_evars->grad_1_B1 + blk_k * BLK_SZ, B1, blk + BLK_ID);
    device::__deriv644_y<pw, pencils, pencil_sz>(
        deriv_evars->grad_1_B2 + blk_k * BLK_SZ, B2, blk + BLK_ID);
    device::__deriv644_y<pw, pencils, pencil_sz>(
        deriv_evars->grad_1_Phi + blk_k * BLK_SZ, Phi, blk + BLK_ID);
    device::__deriv644_y<pw, pencils, pencil_sz>(
        deriv_evars->grad_1_Psi + blk_k * BLK_SZ, Psi, blk + BLK_ID);

    // KO Dissipation
    device::__ko_deriv42_y<pw, pencils, pencil_sz>(
        deriv_evars->ko_grad_1_E0 + blk_k * BLK_SZ, E0, blk + BLK_ID);
    device::__ko_deriv42_y<pw, pencils, pencil_sz>(
        deriv_evars->ko_grad_1_E1 + blk_k * BLK_SZ, E1, blk + BLK_ID);
    device::__ko_deriv42_y<pw, pencils, pencil_sz>(
        deriv_evars->ko_grad_1_E2 + blk_k * BLK_SZ, E2, blk + BLK_ID);
    device::__ko_deriv42_y<pw, pencils, pencil_sz>(
        deriv_evars->ko_grad_1_B0 + blk_k * BLK_SZ, B0, blk + BLK_ID);
    device::__ko_deriv42_y<pw, pencils, pencil_sz>(
        deriv_evars->ko_grad_1_B1 + blk_k * BLK_SZ, B1, blk + BLK_ID);
    device::__ko_deriv42_y<pw, pencils, pencil_sz>(
        deriv_evars->ko_grad_1_B2 + blk_k * BLK_SZ, B2, blk + BLK_ID);
    device::__ko_deriv42_y<pw, pencils, pencil_sz>(
        deriv_evars->ko_grad_1_Phi + blk_k * BLK_SZ, Phi, blk + BLK_ID);
    device::__ko_deriv42_y<pw, pencils, pencil_sz>(
        deriv_evars->ko_grad_1_Psi + blk_k * BLK_SZ, Psi, blk + BLK_ID);
}

template <int pw, int pencils, int pencil_sz, int BATCHED_BLOCKS_SZ>
GLOBAL_FUNC void launch_dir_z_deriv_kernel(
    const device::MeshGPU* const dptr_mesh, const DEVICE_REAL* const u,
    EVAR_DERIVS* deriv_evars, BlockGPU3D* blk, DEVICE_UINT blk_begin) {
    const DEVICE_UINT blk_k  = GPUDevice::block_id_z();
    const DEVICE_UINT BLK_ID = blk_begin + blk_k;
    const DEVICE_UINT BLK_SZ = (blk[BLK_ID].m_aligned_sz[0]) *
                               (blk[BLK_ID].m_aligned_sz[1]) *
                               (blk[BLK_ID].m_aligned_sz[2]);
    const DEVICE_UINT offset     = blk[BLK_ID].m_offset;
    const DEVICE_UINT sz_per_dof = dptr_mesh->m_oct_unzip_sz;

    const DEVICE_REAL* const E0  = &u[dsolve::VAR::U_E0 * sz_per_dof + offset];
    const DEVICE_REAL* const E1  = &u[dsolve::VAR::U_E1 * sz_per_dof + offset];
    const DEVICE_REAL* const E2  = &u[dsolve::VAR::U_E2 * sz_per_dof + offset];
    const DEVICE_REAL* const B0  = &u[dsolve::VAR::U_B0 * sz_per_dof + offset];
    const DEVICE_REAL* const B1  = &u[dsolve::VAR::U_B1 * sz_per_dof + offset];
    const DEVICE_REAL* const B2  = &u[dsolve::VAR::U_B2 * sz_per_dof + offset];
    const DEVICE_REAL* const Phi = &u[dsolve::VAR::U_PHI * sz_per_dof + offset];
    const DEVICE_REAL* const Psi = &u[dsolve::VAR::U_PSI * sz_per_dof + offset];

    device::__deriv644_z<pw, pencils, pencil_sz>(
        deriv_evars->grad_2_E0 + blk_k * BLK_SZ, E0, blk + BLK_ID);
    device::__deriv644_z<pw, pencils, pencil_sz>(
        deriv_evars->grad_2_E1 + blk_k * BLK_SZ, E1, blk + BLK_ID);
    device::__deriv644_z<pw, pencils, pencil_sz>(
        deriv_evars->grad_2_E2 + blk_k * BLK_SZ, E2, blk + BLK_ID);
    device::__deriv644_z<pw, pencils, pencil_sz>(
        deriv_evars->grad_2_B0 + blk_k * BLK_SZ, B0, blk + BLK_ID);
    device::__deriv644_z<pw, pencils, pencil_sz>(
        deriv_evars->grad_2_B1 + blk_k * BLK_SZ, B1, blk + BLK_ID);
    device::__deriv644_z<pw, pencils, pencil_sz>(
        deriv_evars->grad_2_B2 + blk_k * BLK_SZ, B2, blk + BLK_ID);
    device::__deriv644_z<pw, pencils, pencil_sz>(
        deriv_evars->grad_2_Phi + blk_k * BLK_SZ, Phi, blk + BLK_ID);
    device::__deriv644_z<pw, pencils, pencil_sz>(
        deriv_evars->grad_2_Psi + blk_k * BLK_SZ, Psi, blk + BLK_ID);

    // KO Dissipation
    device::__ko_deriv42_z<pw, pencils, pencil_sz>(
        deriv_evars->ko_grad_2_E0 + blk_k * BLK_SZ, E0, blk + BLK_ID);
    device::__ko_deriv42_z<pw, pencils, pencil_sz>(
        deriv_evars->ko_grad_2_E1 + blk_k * BLK_SZ, E1, blk + BLK_ID);
    device::__ko_deriv42_z<pw, pencils, pencil_sz>(
        deriv_evars->ko_grad_2_E2 + blk_k * BLK_SZ, E2, blk + BLK_ID);
    device::__ko_deriv42_z<pw, pencils, pencil_sz>(
        deriv_evars->ko_grad_2_B0 + blk_k * BLK_SZ, B0, blk + BLK_ID);
    device::__ko_deriv42_z<pw, pencils, pencil_sz>(
        deriv_evars->ko_grad_2_B1 + blk_k * BLK_SZ, B1, blk + BLK_ID);
    device::__ko_deriv42_z<pw, pencils, pencil_sz>(
        deriv_evars->ko_grad_2_B2 + blk_k * BLK_SZ, B2, blk + BLK_ID);
    device::__ko_deriv42_z<pw, pencils, pencil_sz>(
        deriv_evars->ko_grad_2_Phi + blk_k * BLK_SZ, Phi, blk + BLK_ID);
    device::__ko_deriv42_z<pw, pencils, pencil_sz>(
        deriv_evars->ko_grad_2_Psi + blk_k * BLK_SZ, Psi, blk + BLK_ID);
}

template <int pw, int pencils, int pencil_sz, int BATCHED_BLOCKS_SZ>
GLOBAL_FUNC void launch_rhs(const device::MeshGPU* const dptr_mesh,
                            DEVICE_REAL* const Fu, const DEVICE_REAL* const u,
                            EVAR_DERIVS* deriv_evars, BlockGPU3D* blk,
                            DEVICE_UINT blk_begin,
                            const SOLVERComputeParams solverParams) {
    const DEVICE_UINT blk_k  = GPUDevice::block_id_z();
    const DEVICE_UINT BLK_ID = blk_begin + blk_k;

    const DEVICE_UINT BLK_SZ = (blk[BLK_ID].m_aligned_sz[0]) *
                               (blk[BLK_ID].m_aligned_sz[1]) *
                               (blk[BLK_ID].m_aligned_sz[2]);
    const DEVICE_UINT offset     = blk[BLK_ID].m_offset;
    const DEVICE_UINT bflag      = blk[BLK_ID].m_bflag;
    const DEVICE_UINT sz_per_dof = dptr_mesh->m_oct_unzip_sz;

    const DEVICE_INT nx          = blk[BLK_ID].m_sz[0];
    const DEVICE_INT ny          = blk[BLK_ID].m_sz[1];
    const DEVICE_INT nz          = blk[BLK_ID].m_sz[2];

    const DEVICE_REAL hx         = blk[BLK_ID].m_dx[0];
    const DEVICE_REAL hy         = blk[BLK_ID].m_dx[1];
    const DEVICE_REAL hz         = blk[BLK_ID].m_dx[2];

    const DEVICE_INT i           = GPUDevice::thread_id_x();
    const DEVICE_INT j = GPUDevice::block_id_x() * GPUDevice::block_dim_y() +
                         GPUDevice::thread_id_y();
    const DEVICE_INT k = GPUDevice::block_id_y();

    const DEVICE_INT pp =
        k * (blk[BLK_ID].m_aligned_sz[0] * blk[BLK_ID].m_aligned_sz[1]) +
        j * blk[BLK_ID].m_aligned_sz[0] + i;

    const DEVICE_REAL x          = blk[BLK_ID].m_ptMin[0] + i * hx;
    const DEVICE_REAL y          = blk[BLK_ID].m_ptMin[1] + j * hy;
    const DEVICE_REAL z          = blk[BLK_ID].m_ptMin[2] + k * hz;

    //  INPUT VARS DEVICE REAL - const Variables
    const DEVICE_REAL* const E0  = &u[dsolve::VAR::U_E0 * sz_per_dof + offset];
    const DEVICE_REAL* const E1  = &u[dsolve::VAR::U_E1 * sz_per_dof + offset];
    const DEVICE_REAL* const E2  = &u[dsolve::VAR::U_E2 * sz_per_dof + offset];
    const DEVICE_REAL* const B0  = &u[dsolve::VAR::U_B0 * sz_per_dof + offset];
    const DEVICE_REAL* const B1  = &u[dsolve::VAR::U_B1 * sz_per_dof + offset];
    const DEVICE_REAL* const B2  = &u[dsolve::VAR::U_B2 * sz_per_dof + offset];
    const DEVICE_REAL* const Phi = &u[dsolve::VAR::U_PHI * sz_per_dof + offset];
    const DEVICE_REAL* const Psi = &u[dsolve::VAR::U_PSI * sz_per_dof + offset];

    // OUTPUT VARS DEVICE REAL
    DEVICE_REAL* const E0_rhs    = &Fu[dsolve::VAR::U_E0 * sz_per_dof + offset];
    DEVICE_REAL* const E1_rhs    = &Fu[dsolve::VAR::U_E1 * sz_per_dof + offset];
    DEVICE_REAL* const E2_rhs    = &Fu[dsolve::VAR::U_E2 * sz_per_dof + offset];
    DEVICE_REAL* const B0_rhs    = &Fu[dsolve::VAR::U_B0 * sz_per_dof + offset];
    DEVICE_REAL* const B1_rhs    = &Fu[dsolve::VAR::U_B1 * sz_per_dof + offset];
    DEVICE_REAL* const B2_rhs    = &Fu[dsolve::VAR::U_B2 * sz_per_dof + offset];
    DEVICE_REAL* const Phi_rhs = &Fu[dsolve::VAR::U_PHI * sz_per_dof + offset];
    DEVICE_REAL* const Psi_rhs = &Fu[dsolve::VAR::U_PSI * sz_per_dof + offset];

    // Normal Derivatives
    const DEVICE_REAL* const grad_0_E0 =
        (deriv_evars->grad_0_E0 + blk_k * BLK_SZ);
    const DEVICE_REAL* const grad_1_E0 =
        (deriv_evars->grad_1_E0 + blk_k * BLK_SZ);
    const DEVICE_REAL* const grad_2_E0 =
        (deriv_evars->grad_2_E0 + blk_k * BLK_SZ);
    const DEVICE_REAL* const grad_0_E1 =
        (deriv_evars->grad_0_E1 + blk_k * BLK_SZ);
    const DEVICE_REAL* const grad_1_E1 =
        (deriv_evars->grad_1_E1 + blk_k * BLK_SZ);
    const DEVICE_REAL* const grad_2_E1 =
        (deriv_evars->grad_2_E1 + blk_k * BLK_SZ);
    const DEVICE_REAL* const grad_0_E2 =
        (deriv_evars->grad_0_E2 + blk_k * BLK_SZ);
    const DEVICE_REAL* const grad_1_E2 =
        (deriv_evars->grad_1_E2 + blk_k * BLK_SZ);
    const DEVICE_REAL* const grad_2_E2 =
        (deriv_evars->grad_2_E2 + blk_k * BLK_SZ);

    const DEVICE_REAL* const grad_0_B0 =
        (deriv_evars->grad_0_B0 + blk_k * BLK_SZ);
    const DEVICE_REAL* const grad_1_B0 =
        (deriv_evars->grad_1_B0 + blk_k * BLK_SZ);
    const DEVICE_REAL* const grad_2_B0 =
        (deriv_evars->grad_2_B0 + blk_k * BLK_SZ);
    const DEVICE_REAL* const grad_0_B1 =
        (deriv_evars->grad_0_B1 + blk_k * BLK_SZ);
    const DEVICE_REAL* const grad_1_B1 =
        (deriv_evars->grad_1_B1 + blk_k * BLK_SZ);
    const DEVICE_REAL* const grad_2_B1 =
        (deriv_evars->grad_2_B1 + blk_k * BLK_SZ);
    const DEVICE_REAL* const grad_0_B2 =
        (deriv_evars->grad_0_B2 + blk_k * BLK_SZ);
    const DEVICE_REAL* const grad_1_B2 =
        (deriv_evars->grad_1_B2 + blk_k * BLK_SZ);
    const DEVICE_REAL* const grad_2_B2 =
        (deriv_evars->grad_2_B2 + blk_k * BLK_SZ);

    const DEVICE_REAL* const grad_0_Phi =
        (deriv_evars->grad_0_Phi + blk_k * BLK_SZ);
    const DEVICE_REAL* const grad_1_Phi =
        (deriv_evars->grad_1_Phi + blk_k * BLK_SZ);
    const DEVICE_REAL* const grad_2_Phi =
        (deriv_evars->grad_2_Phi + blk_k * BLK_SZ);
    const DEVICE_REAL* const grad_0_Psi =
        (deriv_evars->grad_0_Psi + blk_k * BLK_SZ);
    const DEVICE_REAL* const grad_1_Psi =
        (deriv_evars->grad_1_Psi + blk_k * BLK_SZ);
    const DEVICE_REAL* const grad_2_Psi =
        (deriv_evars->grad_2_Psi + blk_k * BLK_SZ);

    const DEVICE_REAL* const ko_grad_0_E0 =
        (deriv_evars->ko_grad_0_E0 + blk_k * BLK_SZ);
    const DEVICE_REAL* const ko_grad_1_E0 =
        (deriv_evars->ko_grad_1_E0 + blk_k * BLK_SZ);
    const DEVICE_REAL* const ko_grad_2_E0 =
        (deriv_evars->ko_grad_2_E0 + blk_k * BLK_SZ);
    const DEVICE_REAL* const ko_grad_0_E1 =
        (deriv_evars->ko_grad_0_E1 + blk_k * BLK_SZ);
    const DEVICE_REAL* const ko_grad_1_E1 =
        (deriv_evars->ko_grad_1_E1 + blk_k * BLK_SZ);
    const DEVICE_REAL* const ko_grad_2_E1 =
        (deriv_evars->ko_grad_2_E1 + blk_k * BLK_SZ);
    const DEVICE_REAL* const ko_grad_0_E2 =
        (deriv_evars->ko_grad_0_E2 + blk_k * BLK_SZ);
    const DEVICE_REAL* const ko_grad_1_E2 =
        (deriv_evars->ko_grad_1_E2 + blk_k * BLK_SZ);
    const DEVICE_REAL* const ko_grad_2_E2 =
        (deriv_evars->ko_grad_2_E2 + blk_k * BLK_SZ);

    const DEVICE_REAL* const ko_grad_0_B0 =
        (deriv_evars->ko_grad_0_B0 + blk_k * BLK_SZ);
    const DEVICE_REAL* const ko_grad_1_B0 =
        (deriv_evars->ko_grad_1_B0 + blk_k * BLK_SZ);
    const DEVICE_REAL* const ko_grad_2_B0 =
        (deriv_evars->ko_grad_2_B0 + blk_k * BLK_SZ);
    const DEVICE_REAL* const ko_grad_0_B1 =
        (deriv_evars->ko_grad_0_B1 + blk_k * BLK_SZ);
    const DEVICE_REAL* const ko_grad_1_B1 =
        (deriv_evars->ko_grad_1_B1 + blk_k * BLK_SZ);
    const DEVICE_REAL* const ko_grad_2_B1 =
        (deriv_evars->ko_grad_2_B1 + blk_k * BLK_SZ);
    const DEVICE_REAL* const ko_grad_0_B2 =
        (deriv_evars->ko_grad_0_B2 + blk_k * BLK_SZ);
    const DEVICE_REAL* const ko_grad_1_B2 =
        (deriv_evars->ko_grad_1_B2 + blk_k * BLK_SZ);
    const DEVICE_REAL* const ko_grad_2_B2 =
        (deriv_evars->ko_grad_2_B2 + blk_k * BLK_SZ);

    const DEVICE_REAL* const ko_grad_0_Phi =
        (deriv_evars->ko_grad_0_Phi + blk_k * BLK_SZ);
    const DEVICE_REAL* const ko_grad_1_Phi =
        (deriv_evars->ko_grad_1_Phi + blk_k * BLK_SZ);
    const DEVICE_REAL* const ko_grad_2_Phi =
        (deriv_evars->ko_grad_2_Phi + blk_k * BLK_SZ);
    const DEVICE_REAL* const ko_grad_0_Psi =
        (deriv_evars->ko_grad_0_Psi + blk_k * BLK_SZ);
    const DEVICE_REAL* const ko_grad_1_Psi =
        (deriv_evars->ko_grad_1_Psi + blk_k * BLK_SZ);
    const DEVICE_REAL* const ko_grad_2_Psi =
        (deriv_evars->ko_grad_2_Psi + blk_k * BLK_SZ);

    const DEVICE_REAL r  = sqrt(x * x + y * y + z * z);

    const double kappa_1 = solverParams.kappa_1;
    const double kappa_2 = solverParams.kappa_2;
    const double sigma   = solverParams.KO_DISS_SIGMA;
    const double rho_e   = 0.0;
    const double J0      = 0.0;
    const double J1      = 0.0;
    const double J2      = 0.0;

    E0_rhs[pp] =
        -4.0 * M_PI * J0 - grad_0_Psi[pp] + grad_1_B2[pp] - grad_2_B1[pp];
    E1_rhs[pp] =
        -4.0 * M_PI * J1 - grad_1_Psi[pp] - grad_0_B2[pp] + grad_2_B0[pp];
    E2_rhs[pp] =
        -4.0 * M_PI * J2 - grad_2_Psi[pp] + grad_0_B1[pp] - grad_1_B0[pp];

    B0_rhs[pp]  = grad_0_Phi[pp] - grad_1_E2[pp] + grad_2_E1[pp];
    B1_rhs[pp]  = grad_1_Phi[pp] + grad_0_E2[pp] - grad_2_E0[pp];
    B2_rhs[pp]  = grad_2_Phi[pp] - grad_0_E1[pp] + grad_1_E0[pp];

    Psi_rhs[pp] = 4.0 * M_PI * rho_e - Psi[pp] * kappa_1 -
                  (grad_0_E0[pp] + grad_1_E1[pp] + grad_2_E2[pp]);
    Phi_rhs[pp] =
        -Phi[pp] * kappa_2 + (grad_0_B0[pp] + grad_1_B1[pp] + grad_2_B2[pp]);

    // Add KO Dissipation
    E0_rhs[pp] +=
        sigma * (ko_grad_0_E0[pp] + ko_grad_1_E0[pp] + ko_grad_2_E0[pp]);
    E1_rhs[pp] +=
        sigma * (ko_grad_0_E1[pp] + ko_grad_1_E1[pp] + ko_grad_2_E1[pp]);
    E2_rhs[pp] +=
        sigma * (ko_grad_0_E2[pp] + ko_grad_1_E2[pp] + ko_grad_2_E2[pp]);
    B0_rhs[pp] +=
        sigma * (ko_grad_0_B0[pp] + ko_grad_1_B0[pp] + ko_grad_2_B0[pp]);
    B1_rhs[pp] +=
        sigma * (ko_grad_0_B1[pp] + ko_grad_1_B1[pp] + ko_grad_2_B1[pp]);
    B2_rhs[pp] +=
        sigma * (ko_grad_0_B2[pp] + ko_grad_1_B2[pp] + ko_grad_2_B2[pp]);
    Phi_rhs[pp] +=
        sigma * (ko_grad_0_Phi[pp] + ko_grad_1_Phi[pp] + ko_grad_2_Phi[pp]);
    Psi_rhs[pp] +=
        sigma * (ko_grad_0_Psi[pp] + ko_grad_1_Psi[pp] + ko_grad_2_Psi[pp]);

    __syncthreads();

    if (bflag != 0) {
        device::radiative_bc<3>(E0_rhs, E0, grad_0_E0, grad_1_E0, grad_2_E0,
                                2.0, 0.0, blk, BLK_ID);
        device::radiative_bc<3>(E1_rhs, E1, grad_0_E1, grad_1_E1, grad_2_E1,
                                2.0, 0.0, blk, BLK_ID);
        device::radiative_bc<3>(E2_rhs, E2, grad_0_E2, grad_1_E2, grad_2_E2,
                                2.0, 0.0, blk, BLK_ID);
        device::radiative_bc<3>(B0_rhs, B0, grad_0_B0, grad_1_B0, grad_2_B0,
                                2.0, 0.0, blk, BLK_ID);
        device::radiative_bc<3>(B1_rhs, B1, grad_0_B1, grad_1_B1, grad_2_B1,
                                2.0, 0.0, blk, BLK_ID);
        device::radiative_bc<3>(B2_rhs, B2, grad_0_B2, grad_1_B2, grad_2_B2,
                                2.0, 0.0, blk, BLK_ID);
        device::radiative_bc<3>(Phi_rhs, Phi, grad_0_Phi, grad_1_Phi,
                                grad_2_Phi, 2.0, 0.0, blk, BLK_ID);
        device::radiative_bc<3>(Psi_rhs, Psi, grad_0_Psi, grad_1_Psi,
                                grad_2_Psi, 2.0, 0.0, blk, BLK_ID);
    }
}

EM4CtxGPU::EM4CtxGPU(ot::Mesh* pMesh)
    : ts::Ctx<EM4CtxGPU, DendroScalar, unsigned int>() {
    m_uiMesh    = pMesh;
    m_mesh_cpu  = device::MeshGPU();
    m_dptr_mesh = m_mesh_cpu.alloc_mesh_on_device(m_uiMesh);

    m_var[VL::CPU_EV].create_vector(m_uiMesh, ot::DVEC_TYPE::OCT_SHARED_NODES,
                                    ot::DVEC_LOC::HOST, dsolve::SOLVER_NUM_VARS,
                                    true);
    m_var[VL::CPU_EV_UZ].create_vector(
        m_uiMesh, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::HOST,
        dsolve::SOLVER_NUM_VARS, true);

    m_var[VL::GPU_EV].create_vector(m_uiMesh, ot::DVEC_TYPE::OCT_SHARED_NODES,
                                    ot::DVEC_LOC::DEVICE,
                                    dsolve::SOLVER_NUM_VARS, true);
    m_var[VL::GPU_EV_UZ_IN].create_vector(
        m_uiMesh, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::DEVICE,
        dsolve::SOLVER_NUM_VARS, true);
    m_var[VL::GPU_EV_UZ_OUT].create_vector(
        m_uiMesh, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::DEVICE,
        dsolve::SOLVER_NUM_VARS, true);

    m_uiTinfo._m_uiStep = 0;
    m_uiTinfo._m_uiT    = 0;
    m_uiTinfo._m_uiTb   = dsolve::SOLVER_RK_TIME_BEGIN;
    m_uiTinfo._m_uiTe   = dsolve::SOLVER_RK_TIME_END;
    m_uiTinfo._m_uiTh   = dsolve::SOLVER_RK45_TIME_STEP_SIZE;

    m_uiElementOrder    = dsolve::SOLVER_ELE_ORDER;

    m_uiMinPt = Point(dsolve::SOLVER_GRID_MIN_X, dsolve::SOLVER_GRID_MIN_Y,
                      dsolve::SOLVER_GRID_MIN_Z);
    m_uiMaxPt = Point(dsolve::SOLVER_GRID_MAX_X, dsolve::SOLVER_GRID_MAX_Y,
                      dsolve::SOLVER_GRID_MAX_Z);

    ot::dealloc_mpi_ctx<DendroScalar>(m_uiMesh, m_mpi_ctx,
                                      dsolve::SOLVER_NUM_VARS,
                                      dsolve::SOLVER_ASYNC_COMM_K);
    ot::alloc_mpi_ctx<DendroScalar>(m_uiMesh, m_mpi_ctx,
                                    dsolve::SOLVER_NUM_VARS,
                                    dsolve::SOLVER_ASYNC_COMM_K);

    device::dealloc_mpi_ctx<DendroScalar>(m_uiMesh, m_mpi_ctx_device,
                                          dsolve::SOLVER_NUM_VARS,
                                          dsolve::SOLVER_ASYNC_COMM_K);
    device::alloc_mpi_ctx<DendroScalar>(m_uiMesh, m_mpi_ctx_device,
                                        dsolve::SOLVER_NUM_VARS,
                                        dsolve::SOLVER_ASYNC_COMM_K);

    const unsigned int PW         = dsolve::SOLVER_PADDING_WIDTH;
    const unsigned int BLK_SZ     = (2 * PW + 7) * (2 * PW + 7) * (2 * PW + 7);

    EVAR_DERIVS* deriv_evars      = GPUDevice::host_malloc<EVAR_DERIVS>(1);
    EVAR_DERIVS* dptr_deriv_evars = GPUDevice::device_malloc<EVAR_DERIVS>(1);

    DEVICE_REAL* deriv_base       = GPUDevice::device_malloc<DEVICE_REAL>(
        48 * DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ);  // 24 normal + 24 KO

    DEVICE_REAL* ptr       = deriv_base;
    deriv_evars->grad_0_E0 = ptr;
    ptr += DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ;
    deriv_evars->grad_1_E0 = ptr;
    ptr += DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ;
    deriv_evars->grad_2_E0 = ptr;
    ptr += DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ;
    deriv_evars->grad_0_E1 = ptr;
    ptr += DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ;
    deriv_evars->grad_1_E1 = ptr;
    ptr += DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ;
    deriv_evars->grad_2_E1 = ptr;
    ptr += DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ;
    deriv_evars->grad_0_E2 = ptr;
    ptr += DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ;
    deriv_evars->grad_1_E2 = ptr;
    ptr += DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ;
    deriv_evars->grad_2_E2 = ptr;
    ptr += DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ;
    deriv_evars->grad_0_B0 = ptr;
    ptr += DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ;
    deriv_evars->grad_1_B0 = ptr;
    ptr += DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ;
    deriv_evars->grad_2_B0 = ptr;
    ptr += DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ;
    deriv_evars->grad_0_B1 = ptr;
    ptr += DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ;
    deriv_evars->grad_1_B1 = ptr;
    ptr += DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ;
    deriv_evars->grad_2_B1 = ptr;
    ptr += DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ;
    deriv_evars->grad_0_B2 = ptr;
    ptr += DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ;
    deriv_evars->grad_1_B2 = ptr;
    ptr += DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ;
    deriv_evars->grad_2_B2 = ptr;
    ptr += DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ;
    deriv_evars->grad_0_Phi = ptr;
    ptr += DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ;
    deriv_evars->grad_1_Phi = ptr;
    ptr += DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ;
    deriv_evars->grad_2_Phi = ptr;
    ptr += DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ;
    deriv_evars->grad_0_Psi = ptr;
    ptr += DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ;
    deriv_evars->grad_1_Psi = ptr;
    ptr += DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ;
    deriv_evars->grad_2_Psi = ptr;
    ptr += DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ;

    deriv_evars->ko_grad_0_E0 = ptr;
    ptr += DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ;
    deriv_evars->ko_grad_1_E0 = ptr;
    ptr += DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ;
    deriv_evars->ko_grad_2_E0 = ptr;
    ptr += DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ;
    deriv_evars->ko_grad_0_E1 = ptr;
    ptr += DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ;
    deriv_evars->ko_grad_1_E1 = ptr;
    ptr += DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ;
    deriv_evars->ko_grad_2_E1 = ptr;
    ptr += DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ;
    deriv_evars->ko_grad_0_E2 = ptr;
    ptr += DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ;
    deriv_evars->ko_grad_1_E2 = ptr;
    ptr += DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ;
    deriv_evars->ko_grad_2_E2 = ptr;
    ptr += DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ;
    deriv_evars->ko_grad_0_B0 = ptr;
    ptr += DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ;
    deriv_evars->ko_grad_1_B0 = ptr;
    ptr += DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ;
    deriv_evars->ko_grad_2_B0 = ptr;
    ptr += DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ;
    deriv_evars->ko_grad_0_B1 = ptr;
    ptr += DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ;
    deriv_evars->ko_grad_1_B1 = ptr;
    ptr += DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ;
    deriv_evars->ko_grad_2_B1 = ptr;
    ptr += DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ;
    deriv_evars->ko_grad_0_B2 = ptr;
    ptr += DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ;
    deriv_evars->ko_grad_1_B2 = ptr;
    ptr += DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ;
    deriv_evars->ko_grad_2_B2 = ptr;
    ptr += DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ;
    deriv_evars->ko_grad_0_Phi = ptr;
    ptr += DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ;
    deriv_evars->ko_grad_1_Phi = ptr;
    ptr += DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ;
    deriv_evars->ko_grad_2_Phi = ptr;
    ptr += DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ;
    deriv_evars->ko_grad_0_Psi = ptr;
    ptr += DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ;
    deriv_evars->ko_grad_1_Psi = ptr;
    ptr += DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ;
    deriv_evars->ko_grad_2_Psi = ptr;
    ptr += DEVICE_RHS_BATCHED_GRAIN_SZ * BLK_SZ;

    GPUDevice::host_to_device(deriv_evars, dptr_deriv_evars, 1);

    m_deriv_evars      = deriv_evars;
    m_dptr_deriv_evars = dptr_deriv_evars;
    m_dptr_deriv_base  = deriv_base;
}

EM4CtxGPU::~EM4CtxGPU() {
    for (unsigned int i = 0; i < VL::END; i++) m_var[i].destroy_vector();
    GPUDevice::device_free(m_dptr_deriv_base);
    GPUDevice::device_free(m_dptr_deriv_evars);
    GPUDevice::host_free(m_deriv_evars);
}

int EM4CtxGPU::host_to_device_sync() {
    if (!m_uiMesh->isActive()) return 0;
    DVec& m_evar      = m_var[VL::CPU_EV];
    DVec& m_dptr_evar = m_var[VL::GPU_EV];

    GPUDevice::host_to_device(m_evar.get_vec_ptr(), m_dptr_evar.get_vec_ptr(),
                              m_evar.get_size());

    return 0;
}

int EM4CtxGPU::device_to_host_sync() {
    if (!m_uiMesh->isActive()) return 0;

    DVec& m_evar      = m_var[VL::CPU_EV];
    DVec& m_dptr_evar = m_var[VL::GPU_EV];
    GPUDevice::device_to_host(m_evar.get_vec_ptr(), m_dptr_evar.get_vec_ptr(),
                              m_evar.get_size());

    return 0;
}

int EM4CtxGPU::initialize() {
    DVec& m_evar      = m_var[VL::CPU_EV];
    DVec& m_dptr_evar = m_var[VL::GPU_EV];

    if (dsolve::SOLVER_RESTORE_SOLVER) {
        this->restore_checkpt();
        this->host_to_device_sync();
        return 0;
    }

    this->init_grid();

    bool isRefine = false;
    DendroIntL oldElements, oldElements_g;
    DendroIntL newElements, newElements_g;
    DendroIntL oldGridPoints, oldGridPoints_g;
    DendroIntL newGridPoints, newGridPoints_g;

    unsigned int iterCount         = 1;
    const unsigned int max_iter    = dsolve::SOLVER_INIT_GRID_ITER;
    const unsigned int rank_global = m_uiMesh->getMPIRankGlobal();
    MPI_Comm gcomm                 = m_uiMesh->getMPIGlobalCommunicator();

    DendroScalar* unzipVar[dsolve::SOLVER_NUM_VARS];
    unsigned int refineVarIds[dsolve::SOLVER_NUM_REFINE_VARS];

    for (unsigned int vIndex = 0; vIndex < dsolve::SOLVER_NUM_REFINE_VARS;
         vIndex++)
        refineVarIds[vIndex] = dsolve::SOLVER_REFINE_VARIABLE_INDICES[vIndex];

    std::function<double(double, double, double, double* hx)> waveletTolFunc =
        [](double x, double y, double z, double* hx) {
            return dsolve::computeWTolDCoords(x, y, z, hx);
        };

    DVec& m_evar_unz = m_var[VL::CPU_EV_UZ];

    do {
        // Re-initialize grid at each step of convergence
        this->init_grid();

        this->unzip(m_evar, m_evar_unz, dsolve::SOLVER_ASYNC_COMM_K);
        m_evar_unz.to_2d(unzipVar);

        if (max_iter == 0) {
            isRefine = false;
        } else {
            isRefine = m_uiMesh->isReMeshUnzip(
                (const double**)unzipVar, refineVarIds,
                dsolve::SOLVER_NUM_REFINE_VARS, waveletTolFunc,
                dsolve::SOLVER_DENDRO_AMR_FAC);
        }

        if (isRefine) {
            ot::Mesh* newMesh = this->remesh(dsolve::SOLVER_DENDRO_GRAIN_SZ,
                                             dsolve::SOLVER_LOAD_IMB_TOL,
                                             dsolve::SOLVER_SPLIT_FIX);

            oldElements       = m_uiMesh->getNumLocalMeshElements();
            newElements       = newMesh->getNumLocalMeshElements();
            oldGridPoints     = m_uiMesh->getNumLocalMeshNodes();
            newGridPoints     = newMesh->getNumLocalMeshNodes();

            par::Mpi_Allreduce(&oldElements, &oldElements_g, 1, MPI_SUM, gcomm);
            par::Mpi_Allreduce(&newElements, &newElements_g, 1, MPI_SUM, gcomm);
            par::Mpi_Allreduce(&oldGridPoints, &oldGridPoints_g, 1, MPI_SUM,
                               gcomm);
            par::Mpi_Allreduce(&newGridPoints, &newGridPoints_g, 1, MPI_SUM,
                               gcomm);

            if (!rank_global) {
                std::cout << "[EM4CtxGPU] iter : " << iterCount
                          << " (Remesh triggered) ->  old mesh : "
                          << oldElements_g << " new mesh : " << newElements_g
                          << std::endl;
            }

            this->grid_transfer(newMesh);
            std::swap(m_uiMesh, newMesh);
            delete newMesh;

            m_uiGlobalMeshElements = newElements_g;
            m_uiGlobalGridPoints   = newGridPoints_g;
        }

        iterCount += 1;

    } while (isRefine &&
             (newElements_g != oldElements_g ||
              newGridPoints_g != oldGridPoints_g) &&
             (iterCount < max_iter));

    this->init_grid();

    unsigned int lmin, lmax;
    m_uiMesh->computeMinMaxLevel(lmin, lmax);
    dsolve::SOLVER_RK45_TIME_STEP_SIZE =
        dsolve::SOLVER_CFL_FACTOR *
        ((dsolve::SOLVER_COMPD_MAX[0] - dsolve::SOLVER_COMPD_MIN[0]) *
         ((1u << (m_uiMaxDepth - lmax)) / ((double)dsolve::SOLVER_ELE_ORDER)) /
         ((double)(1u << (m_uiMaxDepth))));
    m_uiTinfo._m_uiTh = dsolve::SOLVER_RK45_TIME_STEP_SIZE;

    this->host_to_device_sync();
    return 0;
}

int EM4CtxGPU::init_grid() {
    DVec& m_evar               = m_var[VL::CPU_EV];
    const ot::TreeNode* pNodes = &(*(m_uiMesh->getAllElements().begin()));
    unsigned int eleOrder      = m_uiMesh->getElementOrder();
    const unsigned int* e2n_cg = &(*(m_uiMesh->getE2NMapping().begin()));
    const unsigned int* e2n_dg = &(*(m_uiMesh->getE2NMapping_DG().begin()));
    const unsigned int nPe     = m_uiMesh->getNumNodesPerElement();
    const unsigned int nodeLocalBegin = m_uiMesh->getNodeLocalBegin();
    const unsigned int nodeLocalEnd   = m_uiMesh->getNodeLocalEnd();

    DendroScalar* zipIn[dsolve::SOLVER_NUM_VARS];
    m_evar.to_2d(zipIn);

#pragma omp parallel for
    for (unsigned int elem = m_uiMesh->getElementLocalBegin();
         elem < m_uiMesh->getElementLocalEnd(); elem++) {
        DendroScalar var[dsolve::SOLVER_NUM_VARS];
        for (unsigned int k = 0; k < (eleOrder + 1); k++)
            for (unsigned int j = 0; j < (eleOrder + 1); j++)
                for (unsigned int i = 0; i < (eleOrder + 1); i++) {
                    const unsigned int nodeLookUp_CG =
                        e2n_cg[elem * nPe +
                               k * (eleOrder + 1) * (eleOrder + 1) +
                               j * (eleOrder + 1) + i];
                    if (nodeLookUp_CG >= nodeLocalBegin &&
                        nodeLookUp_CG < nodeLocalEnd) {
                        const unsigned int nodeLookUp_DG =
                            e2n_dg[elem * nPe +
                                   k * (eleOrder + 1) * (eleOrder + 1) +
                                   j * (eleOrder + 1) + i];
                        unsigned int ownerID, ii_x, jj_y, kk_z;
                        m_uiMesh->dg2eijk(nodeLookUp_DG, ownerID, ii_x, jj_y,
                                          kk_z);
                        const double len =
                            (double)(1u << (m_uiMaxDepth -
                                            pNodes[ownerID].getLevel()));
                        const double x =
                            pNodes[ownerID].getX() + ii_x * (len / (eleOrder));
                        const double y =
                            pNodes[ownerID].getY() + jj_y * (len / (eleOrder));
                        const double z =
                            pNodes[ownerID].getZ() + kk_z * (len / (eleOrder));

                        dsolve::initDataFuncToPhysCoords((double)x, (double)y,
                                                         (double)z, var);

                        for (unsigned int v = 0; v < dsolve::SOLVER_NUM_VARS;
                             v++)
                            zipIn[v][nodeLookUp_CG] = var[v];
                    }
                }
    }

    double h_min[3], h_max[3], o_min[3], o_max[3];
    for (int i = 0; i < 3; ++i) {
        h_min[i] = dsolve::SOLVER_COMPD_MIN[i];
        h_max[i] = dsolve::SOLVER_COMPD_MAX[i];
        o_min[i] = dsolve::SOLVER_OCTREE_MIN[i];
        o_max[i] = dsolve::SOLVER_OCTREE_MAX[i];
    }
    cudaMemcpyToSymbol(cuda::_em4::__SOLVER_COMPD_MIN, h_min,
                       3 * sizeof(double));
    cudaMemcpyToSymbol(cuda::_em4::__SOLVER_COMPD_MAX, h_max,
                       3 * sizeof(double));
    cudaMemcpyToSymbol(cuda::_em4::__SOLVER_OCTREE_MIN, o_min,
                       3 * sizeof(double));
    cudaMemcpyToSymbol(cuda::_em4::__SOLVER_OCTREE_MAX, o_max,
                       3 * sizeof(double));

    return 0;
}

int EM4CtxGPU::rhs(DVec* in, DVec* out, unsigned int sz, DendroScalar time) {
    if (!m_uiMesh->isActive()) return 0;

    const std::vector<ot::Block>& blk_list = m_uiMesh->getLocalBlockList();
    const unsigned int nblocks             = blk_list.size();
    if (nblocks == 0) return 0;

    const unsigned int NUM_BATCHES = nblocks / DEVICE_RHS_BATCHED_GRAIN_SZ + 1;

    const unsigned int nx          = blk_list[0].getAllocationSzX();
    const unsigned int ny          = blk_list[0].getAllocationSzY();
    const unsigned int nz          = blk_list[0].getAllocationSzZ();

    const unsigned int pencils     = 13;
    const unsigned int pen_sz      = 13;

    DVec& m_dptr_uz_i              = m_var[VL::GPU_EV_UZ_IN];
    DVec& m_dptr_uz_o              = m_var[VL::GPU_EV_UZ_OUT];

    this->unzip(*in, m_dptr_uz_i, dsolve::SOLVER_ASYNC_COMM_K);

    SOLVERComputeParams params;
    params.kappa_1       = 0.1;  // dsolve::SOLVER_KAPPA1; // TODO: verify names
    params.kappa_2       = 0.1;  // dsolve::SOLVER_KAPPA2;
    params.KO_DISS_SIGMA = dsolve::KO_DISS_SIGMA;

    for (unsigned int bid = 0; bid < NUM_BATCHES; bid++) {
        const unsigned int block_begin = (bid * nblocks) / NUM_BATCHES;
        const unsigned int block_end   = ((bid + 1) * nblocks) / NUM_BATCHES;
        const unsigned int numblocks   = block_end - block_begin;
        if (numblocks == 0) continue;

        dim3 grid_x  = dim3(ny / pencils, nz, numblocks);
        dim3 block_x = dim3(nx, pencils, 1);

        dim3 grid_y  = dim3(nx / pencils, nz, numblocks);
        dim3 block_y = dim3(pencils, ny, 1);

        dim3 grid_z  = dim3(nx / pencils, ny, numblocks);
        dim3 block_z = dim3(pencils, nz, 1);

        launch_dir_x_deriv_kernel<3, pencils, pen_sz,
                                  DEVICE_RHS_BATCHED_GRAIN_SZ>
            <<<grid_x, block_x, 0, 0>>>(m_dptr_mesh, m_dptr_uz_i.get_vec_ptr(),
                                        m_dptr_deriv_evars,
                                        m_mesh_cpu.m_blk_list, block_begin);

        launch_dir_y_deriv_kernel<3, pencils, pen_sz,
                                  DEVICE_RHS_BATCHED_GRAIN_SZ>
            <<<grid_y, block_y, 0, 0>>>(m_dptr_mesh, m_dptr_uz_i.get_vec_ptr(),
                                        m_dptr_deriv_evars,
                                        m_mesh_cpu.m_blk_list, block_begin);

        launch_dir_z_deriv_kernel<3, pencils, pen_sz,
                                  DEVICE_RHS_BATCHED_GRAIN_SZ>
            <<<grid_z, block_z, 0, 0>>>(m_dptr_mesh, m_dptr_uz_i.get_vec_ptr(),
                                        m_dptr_deriv_evars,
                                        m_mesh_cpu.m_blk_list, block_begin);

        dim3 grid_rhs  = dim3(nx / 32 + 1, ny / 8 + 1, numblocks);
        dim3 block_rhs = dim3(32, 8, 1);

        launch_rhs<3, pencils, pen_sz, DEVICE_RHS_BATCHED_GRAIN_SZ>
            <<<grid_x, block_x, 0, 0>>>(
                m_dptr_mesh, m_dptr_uz_o.get_vec_ptr(),
                m_dptr_uz_i.get_vec_ptr(), m_dptr_deriv_evars,
                m_mesh_cpu.m_blk_list, block_begin, params);

        GPUDevice::check_last_error();
        GPUDevice::device_synchronize();
    }

    this->zip(m_dptr_uz_o, *out);
    return 0;
}

bool EM4CtxGPU::is_remesh() {
    bool isRefine      = false;
    DVec& m_evar       = m_var[VL::CPU_EV];
    DVec& m_evar_unzip = m_var[VL::CPU_EV_UZ];

    this->unzip(m_evar, m_evar_unzip, dsolve::SOLVER_ASYNC_COMM_K);

    DendroScalar* unzipVar[dsolve::SOLVER_NUM_VARS];
    m_evar_unzip.to_2d(unzipVar);

    unsigned int refineVarIds[8];
    for (unsigned int vIndex = 0; vIndex < dsolve::SOLVER_NUM_REFINE_VARS;
         vIndex++)
        refineVarIds[vIndex] = dsolve::SOLVER_REFINE_VARIABLE_INDICES[vIndex];

    std::function<double(double, double, double, double* hx)> waveletTolFunc =
        [](double x, double y, double z, double* hx) {
            return dsolve::computeWTolDCoords(x, y, z, hx);
        };

    isRefine = m_uiMesh->isReMeshUnzip(
        (const double**)unzipVar, refineVarIds, dsolve::SOLVER_NUM_REFINE_VARS,
        waveletTolFunc, dsolve::SOLVER_DENDRO_AMR_FAC);

    return isRefine;
}

int EM4CtxGPU::write_vtu() {
    if (!m_uiMesh->isActive()) return 0;

    DVec& m_evar = m_var[VL::CPU_EV];
    m_uiMesh->readFromGhostBegin(m_evar.get_vec_ptr(), m_evar.get_dof());
    m_uiMesh->readFromGhostEnd(m_evar.get_vec_ptr(), m_evar.get_dof());

    DendroScalar* evolVar[dsolve::SOLVER_NUM_VARS];
    m_evar.to_2d(evolVar);

    std::vector<std::string> pDataNames;
    const unsigned int numConstVars = 0;
    const unsigned int numEvolVars  = dsolve::SOLVER_NUM_EVOL_VARS_VTU_OUTPUT;
    double* pData[numEvolVars];

    for (unsigned int i = 0; i < numEvolVars; i++) {
        pDataNames.push_back(
            std::string(dsolve::SOLVER_VAR_NAMES
                            [dsolve::SOLVER_VTU_OUTPUT_EVOL_INDICES[i]]));
        pData[i] = evolVar[dsolve::SOLVER_VTU_OUTPUT_EVOL_INDICES[i]];
    }

    std::vector<char*> pDataNames_char;
    pDataNames_char.reserve(pDataNames.size());
    for (unsigned int i = 0; i < pDataNames.size(); i++)
        pDataNames_char.push_back(const_cast<char*>(pDataNames[i].c_str()));

    const char* fDataNames[] = {"Time", "Cycle"};
    const double fData[]     = {m_uiTinfo._m_uiT, (double)m_uiTinfo._m_uiStep};

    char fPrefix[256];
    sprintf(fPrefix, "%s_%d", dsolve::SOLVER_VTU_FILE_PREFIX.c_str(),
            m_uiTinfo._m_uiStep);

    if (dsolve::SOLVER_VTU_Z_SLICE_ONLY) {
        unsigned int s_val[3]  = {1u << (m_uiMaxDepth - 1),
                                  1u << (m_uiMaxDepth - 1),
                                  1u << (m_uiMaxDepth - 1)};
        unsigned int s_norm[3] = {0, 0, 1};
        io::vtk::mesh2vtu_slice(m_uiMesh, s_val, s_norm, fPrefix, 2, fDataNames,
                                fData, (numEvolVars + numConstVars),
                                (const char**)&pDataNames_char[0],
                                (const double**)pData);
    } else {
        io::vtk::mesh2vtuFine(m_uiMesh, fPrefix, 2, fDataNames, fData,
                              (numEvolVars + numConstVars),
                              (const char**)&pDataNames_char[0],
                              (const double**)pData);
    }
    return 0;
}
int EM4CtxGPU::write_checkpt() {
    DVec& m_evar = m_var[VL::CPU_EV];
    if (m_uiMesh->isActive()) {
        unsigned int cpIndex =
            (m_uiTinfo._m_uiStep % (2 * dsolve::SOLVER_CHECKPT_FREQ) == 0) ? 0
                                                                           : 1;
        unsigned int rank = m_uiMesh->getMPIRank();

        char fName[256];
        const ot::TreeNode* pNodes = &(*(m_uiMesh->getAllElements().begin() +
                                         m_uiMesh->getElementLocalBegin()));
        sprintf(fName, "%s_octree_%d_%d.oct",
                dsolve::SOLVER_CHKPT_FILE_PREFIX.c_str(), cpIndex, rank);
        io::checkpoint::writeOctToFile(fName, pNodes,
                                       m_uiMesh->getNumLocalMeshElements());

        const unsigned int dof = m_evar.get_dof();
        DendroScalar* eVar[dof];
        m_evar.to_2d(eVar);

        sprintf(fName, "%s_%d_%d.var", dsolve::SOLVER_CHKPT_FILE_PREFIX.c_str(),
                cpIndex, rank);
        io::checkpoint::writeVecToFile(fName, m_uiMesh, (const double**)eVar,
                                       dsolve::SOLVER_NUM_VARS);

        if (!rank) {
            sprintf(fName, "%s_step_%d.cp",
                    dsolve::SOLVER_CHKPT_FILE_PREFIX.c_str(), cpIndex);
            std::ofstream outfile(fName);
            if (!outfile) {
                std::cout << fName << " file open failed " << std::endl;
                return 0;
            }

            json checkPoint;
            checkPoint["DENDRO_TS_TIME_BEGIN"]     = m_uiTinfo._m_uiTb;
            checkPoint["DENDRO_TS_TIME_END"]       = m_uiTinfo._m_uiTe;
            checkPoint["DENDRO_TS_ELEMENT_ORDER"]  = m_uiElementOrder;
            checkPoint["DENDRO_TS_TIME_CURRENT"]   = m_uiTinfo._m_uiT;
            checkPoint["DENDRO_TS_STEP_CURRENT"]   = m_uiTinfo._m_uiStep;
            checkPoint["DENDRO_TS_TIME_STEP_SIZE"] = m_uiTinfo._m_uiTh;
            checkPoint["DENDRO_TS_WAVELET_TOLERANCE"] =
                dsolve::SOLVER_WAVELET_TOL;
            checkPoint["DENDRO_TS_LOAD_IMB_TOLERANCE"] =
                dsolve::SOLVER_LOAD_IMB_TOL;
            checkPoint["DENDRO_TS_NUM_VARS"]       = dsolve::SOLVER_NUM_VARS;
            checkPoint["DENDRO_TS_ACTIVE_COMM_SZ"] = m_uiMesh->getMPICommSize();

            outfile << std::setw(4) << checkPoint << std::endl;
            outfile.close();
        }
    }
    return 0;
}
int EM4CtxGPU::restore_checkpt() { return 0; }
int EM4CtxGPU::post_stage(DVec& sIn) { return 0; }
int EM4CtxGPU::post_timestep(DVec& sIn) { return 0; }

int EM4CtxGPU::terminal_output() {
    if (m_uiMesh->isActive()) {
        DVec& m_evar = m_var[VL::CPU_EV];
        DendroScalar* zippedUp[dsolve::SOLVER_NUM_VARS];
        m_evar.to_2d(zippedUp);

        std::cout << std::scientific;
        std::cout.precision(7);

        // Replace SOLVER_NUM_CONSOLE_OUTPUT_VARS with however you defined it in
        // EM4
        for (unsigned int i = 0; i < dsolve::SOLVER_NUM_CONSOLE_OUTPUT_VARS;
             i++) {
            unsigned int v = dsolve::SOLVER_CONSOLE_OUTPUT_VARS[i];
            double l_min   = vecMin(&zippedUp[v][m_uiMesh->getNodeLocalBegin()],
                                    m_uiMesh->getNumLocalMeshNodes(),
                                    m_uiMesh->getMPICommunicator());
            double l_max   = vecMax(&zippedUp[v][m_uiMesh->getNodeLocalBegin()],
                                    m_uiMesh->getNumLocalMeshNodes(),
                                    m_uiMesh->getMPICommunicator());
            double l2_norm = normL2(&zippedUp[v][m_uiMesh->getNodeLocalBegin()],
                                    m_uiMesh->getNumLocalMeshNodes(),
                                    m_uiMesh->getMPICommunicator());

            if (!(m_uiMesh->getMPIRank())) {
                std::cout << "\t[var]:  " << std::setw(12)
                          << dsolve::SOLVER_VAR_NAMES[v];
                std::cout << " (min, max, l2) : \t ( " << l_min << ", " << l_max
                          << ", " << l2_norm << ") " << std::endl;
            }
        }
    }
    return 0;
}

int EM4CtxGPU::grid_transfer(const ot::Mesh* m_new) {
#ifdef __PROFILE_CTX__
    m_uiCtxpt[ts::CTXPROFILE::GRID_TRASFER].start();
#endif

#ifdef EM4_ENABLE_PROFILING
    dsolve::timer::t_gridTransfer.start();
#endif

    DVec& m_evar = m_var[VL::CPU_EV];
    DVec::grid_transfer(m_uiMesh, m_new, m_evar);

    ot::dealloc_mpi_ctx<DendroScalar>(m_uiMesh, m_mpi_ctx,
                                      dsolve::SOLVER_NUM_VARS,
                                      dsolve::SOLVER_ASYNC_COMM_K);
    ot::alloc_mpi_ctx<DendroScalar>(m_new, m_mpi_ctx, dsolve::SOLVER_NUM_VARS,
                                    dsolve::SOLVER_ASYNC_COMM_K);

    device::dealloc_mpi_ctx<DendroScalar>(m_uiMesh, m_mpi_ctx_device,
                                          dsolve::SOLVER_NUM_VARS,
                                          dsolve::SOLVER_ASYNC_COMM_K);
    device::alloc_mpi_ctx<DendroScalar>(m_new, m_mpi_ctx_device,
                                        dsolve::SOLVER_NUM_VARS,
                                        dsolve::SOLVER_ASYNC_COMM_K);
    // printf("igt ended\n");

    // m_var[VL::CPU_EV_DG].destroy_vector();
    m_var[VL::CPU_EV_UZ].destroy_vector();

    // m_var[VL::GPU_EV_DG].destroy_vector();
    m_var[VL::GPU_EV].destroy_vector();
    m_var[VL::GPU_EV_UZ_IN].destroy_vector();
    m_var[VL::GPU_EV_UZ_OUT].destroy_vector();

    m_var[VL::CPU_EV_UZ].create_vector(
        m_new, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::HOST,
        dsolve::SOLVER_NUM_VARS, true);
    m_var[VL::GPU_EV].create_vector(m_new, ot::DVEC_TYPE::OCT_SHARED_NODES,
                                    ot::DVEC_LOC::DEVICE,
                                    dsolve::SOLVER_NUM_VARS, true);
    m_var[VL::GPU_EV_UZ_IN].create_vector(
        m_new, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::DEVICE,
        dsolve::SOLVER_NUM_VARS, true);
    m_var[VL::GPU_EV_UZ_OUT].create_vector(
        m_new, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::DEVICE,
        dsolve::SOLVER_NUM_VARS, true);

#ifdef __CUDACC__
    m_mesh_cpu.dealloc_mesh_on_device(m_dptr_mesh);
    m_dptr_mesh = m_mesh_cpu.alloc_mesh_on_device(m_new);
#endif

    this->host_to_device_sync();
    // printf("hto d ended\n");
    m_uiIsETSSynced = false;

#ifdef EM4_ENABLE_PROFILING
    dsolve::timer::t_gridTransfer.stop();
#endif

#ifdef __PROFILE_CTX__
    m_uiCtxpt[ts::CTXPROFILE::GRID_TRASFER].stop();
#endif
    return 0;
}

void EM4CtxGPU::calculate_full_grid_size() {
    if (m_uiMesh->isActive()) {
        DendroIntL mesh_elements = m_uiMesh->getNumLocalMeshElements();
        DendroIntL grid_points   = m_uiMesh->getNumLocalMeshNodes();

        par::Mpi_Reduce(&mesh_elements, &m_uiGlobalMeshElements, 1, MPI_SUM, 0,
                        m_uiMesh->getMPICommunicator());
        par::Mpi_Reduce(&grid_points, &m_uiGlobalGridPoints, 1, MPI_SUM, 0,
                        m_uiMesh->getMPICommunicator());
    }
}

void EM4CtxGPU::write_grid_summary_data() {
    if (m_uiMesh->isActive()) {
        if (!m_uiMesh->getMPIRankGlobal()) {
            std::string fname =
                dsolve::SOLVER_PROFILE_FILE_PREFIX + "_GridInfo.dat";
            try {
                std::ofstream file_grid_data;
                file_grid_data.open(fname, std::ofstream::app);
                file_grid_data.precision(12);
                file_grid_data << std::scientific;

                if (!m_uiWroteGridInfoHeader) {
                    file_grid_data << "timeStep,simTime,commSize,wTime,"
                                      "meshSize,totalGridPoints,stepSize\n";
                    m_uiWroteGridInfoHeader = true;
                }

                file_grid_data << m_uiTinfo._m_uiStep << "," << m_uiTinfo._m_uiT
                               << "," << m_uiMesh->getMPICommSize() << ","
                               << MPI_Wtime() << "," << m_uiGlobalMeshElements
                               << "," << m_uiGlobalGridPoints << ","
                               << m_uiTinfo._m_uiTh << "\n";
                file_grid_data.close();
            } catch (const std::exception& e) {
                std::cout << "Error occured while writing grid summary data!"
                          << std::endl;
            }
        }
    }
}
int EM4CtxGPU::finalize() { return 0; }

DVec& EM4CtxGPU::get_evolution_vars() { return m_var[GPU_EV]; }
DVec& EM4CtxGPU::get_evolution_vars_cpu() { return m_var[CPU_EV]; }

}  // namespace cuda
