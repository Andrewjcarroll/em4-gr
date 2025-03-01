/**
 * @file solverCtx.h
 * @author Milinda Fernando
 * @author David Van Komen
 * @brief Application context class for solving the Einstein equations in SOLVER
 * formulation.
 * @version 0.1
 * @date 2019-12-20
 *
 * @copyright Copyright (c) 2019, University of Utah.
 *
 */

#pragma once
#include <iostream>

#include "checkPoint.h"
#include "ctx.h"
#include "dataUtils.h"
#include "derivs.h"
#include "grDef.h"
#include "grUtils.h"
#include "mathMeshUtils.h"
#include "oct2vtk.h"
#include "parUtils.h"
#include "parameters.h"
#include "physcon.h"
#include "rhs.h"
#include "system_constraints.h"

namespace dsolve {

/**@brief smoothing modes avail for LTS recomended for LTS time stepping. */
enum LTS_SMOOTH_MODE { KO = 0, WEIGHT_FUNC };

enum VL {
    CPU_EV = 0,
    CPU_CV,
    CPU_ANALYTIC,
    CPU_ANALYTIC_DIFF,
    CPU_EV_UZ_IN,
    CPU_EV_UZ_OUT,
    CPU_CV_UZ_IN,
    GPU_EV,
    GPU_EV_UZ_IN,
    GPU_EV_UZ_OUT,
#if 0
    // redefine these if analytic needs to be done on a block-wise basis!
    CPU_ANALYTIC_UZ_IN,
    CPU_ANALYTIC_DIFF_UZ_IN,
#endif
    END
};

typedef ot::DVector<DendroScalar, unsigned int> DVec;

class SOLVERCtx : public ts::Ctx<SOLVERCtx, DendroScalar, unsigned int> {
   protected:
    /**@brief: evolution var (zip)*/
    DVec m_var[VL::END];

    /** @brief: Lets us know if we need to compute the constraints for this
     * timestep */
    bool m_constraintsComputed = false;

    /** @brief: Lets us know if we need to compute the analytical for this
     * timestep */
    bool m_analyticalComputed = false;

   public:
    /**@brief: default constructor*/
    SOLVERCtx(ot::Mesh *pMesh);

    /**@brief: default deconstructor*/
    ~SOLVERCtx();

    /** @brief: Any flags that need to be adjusted and updated for the next step
     *
     * This is particularly useful for if the analytical solution or constraints
     * need to be computed for multiple potential reasons, but the reasons
     * aren't necessarily triggered every time step!
     */
    void resetForNextStep() {
        m_analyticalComputed = false;
        m_constraintsComputed = false;
    }

    /**
     * @brief sets time adaptive offset
     * @param tadapoffst
     */
    void set_time_adap_offset(unsigned int tadapoffst) {
        SOLVER_LTS_TS_OFFSET = tadapoffst;
    }

    /**@brief : returns the time adaptive offset value*/
    unsigned int get_time_adap_offset() { return SOLVER_LTS_TS_OFFSET; }

    /**@brief: initial solution and grid convergence calls init_grid()*/
    int initialize();

    /**@brief: initialize the grid, solution. */
    int init_grid();

    /**
     * @brief computes the SOLVER rhs
     *
     * @param in : zipped input
     * @param out : zipped output
     * @param sz  : number of variables.
     * @param time : current time.
     * @return int : status. (0) on success.
     */
    int rhs(DVec *in, DVec *out, unsigned int sz, DendroScalar time);

    /**
     * @brief Compute the analytical solution to the system
     */
    void compute_analytical();

    /**
     * @brief Compute the constraints for the system
     */
    void compute_constraints();

    /**
     * @brief block wise RHS.
     *
     * @param in : input vector (unzip version)
     * @param out : output vector (unzip version)
     * @param blkIDs : local block ids where the rhs is computed.
     * @param sz : size of the block ids
     * @param blk_time : block time  corresponding to the block ids.
     * @return int
     */
    int rhs_blkwise(DVec in, DVec out, const unsigned int *const blkIDs,
                    unsigned int numIds, DendroScalar *blk_time) const;

    int rhs_blk(const DendroScalar *in, DendroScalar *out, unsigned int dof,
                unsigned int local_blk_id, DendroScalar blk_time) const;

    int pre_stage_blk(DendroScalar *in, unsigned int dof,
                      unsigned int local_blk_id, DendroScalar blk_time) const;

    int post_stage_blk(DendroScalar *in, unsigned int dof,
                       unsigned int local_blk_id, DendroScalar blk_time) const;

    int pre_timestep_blk(DendroScalar *in, unsigned int dof,
                         unsigned int local_blk_id,
                         DendroScalar blk_time) const;

    int post_timestep_blk(DendroScalar *in, unsigned int dof,
                          unsigned int local_blk_id,
                          DendroScalar blk_time) const;

    /**@brief: function execute before each stage
     * @param sIn: stage var in.
     */
    int pre_stage(DVec &sIn);

    /**@brief: function execute after each stage
     * @param sIn: stage var in.
     */
    int post_stage(DVec &sIn);

    /**@brief: function execute before each step*/
    int pre_timestep(DVec &sIn);

    /**@brief: function execute after each step*/
    int post_timestep(DVec &sIn);

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
    DVec &get_evolution_vars();

    /**@brief: pack and returns the constraint variables to one DVector*/
    DVec &get_constraint_vars();

    /**@brief: pack and returns the primitive variables to one DVector*/
    DVec &get_primitive_vars();

    /**@brief: prints any messages to the terminal output. */
    int terminal_output();

    /**@brief: returns the async communication batch size. */
    unsigned int get_async_batch_sz() { return dsolve::SOLVER_ASYNC_COMM_K; }

    /**@brief: returns the number of variables considered when performing
     * refinement*/
    unsigned int get_num_refine_vars() { return SOLVER_NUM_REFINE_VARS; }

    /**@brief: return the pointer for containing evolution refinement variable
     * ids*/
    const unsigned int *get_refine_var_ids() {
        return SOLVER_REFINE_VARIABLE_INDICES;
    }

    /**@brief return the wavelet tolerance function / value*/
    std::function<double(double, double, double, double *hx)>
    get_wtol_function() {
        double wtol = SOLVER_WAVELET_TOL;
        std::function<double(double, double, double, double *)> waveletTolFunc =
            [](double x, double y, double z, double *hx) {
                return dsolve::computeWTolDCoords(x, y, z, hx);
            };
        return waveletTolFunc;
    }

    /**@brief computes the LTS TS offset based on the eta damping.*/
    unsigned int compute_lts_ts_offset();

    /**@brief : blk time step factor. */
    static unsigned int getBlkTimestepFac(unsigned int blev, unsigned int lmin,
                                          unsigned int lmax);

    /**
     * @brief LTS smooth mode.
     *
     * @param sIn : time synced evolution vector.
     * @param mode : smoothing mode.
     */
    void lts_smooth(DVec sIn, LTS_SMOOTH_MODE mode);

    int grid_transfer(const ot::Mesh *m_new);
};

}  // end of namespace dsolve
