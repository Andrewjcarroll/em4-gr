/**
 * @file solverCtx.cpp
 * @author Milinda Fernando (milinda@cs.utah.edu)
 * @brief SOLVER contex file.
 * @version 0.1
 * @date 2019-12-20
 *
 * School of Computing, University of Utah.
 * @copyright Copyright (c) 2019
 *
 */

#include "solverCtx.h"

#include <stdlib.h>

#include <ios>

#include "dendro.h"
#include "derivs.h"
#include "grDef.h"
#include "grUtils.h"
#include "parameters.h"
#include "profile_params.h"

#ifdef EM4_ENABLE_COMPACT_DERIVS
#include "compact_derivs.h"
#endif

namespace dsolve {
SOLVERCtx::SOLVERCtx(ot::Mesh *pMesh) : Ctx() {
    m_uiMesh = pMesh;

    // variable declaration
    m_var[VL::CPU_EV].create_vector(m_uiMesh, ot::DVEC_TYPE::OCT_SHARED_NODES,
                                    ot::DVEC_LOC::HOST, SOLVER_NUM_VARS, true);

    m_var[VL::CPU_EV_UZ_IN].create_vector(
        m_uiMesh, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::HOST,
        SOLVER_NUM_VARS, true);
    m_var[VL::CPU_EV_UZ_OUT].create_vector(
        m_uiMesh, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::HOST,
        SOLVER_NUM_VARS, true);

    m_var[VL::CPU_CV].create_vector(m_uiMesh, ot::DVEC_TYPE::OCT_SHARED_NODES,
                                    ot::DVEC_LOC::HOST,
                                    SOLVER_CONSTRAINT_NUM_VARS, true);
    m_var[VL::CPU_CV_UZ_IN].create_vector(
        m_uiMesh, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::HOST,
        SOLVER_CONSTRAINT_NUM_VARS, true);

    // ADDED TTO COMPARE TO THE ANAYLYTIC SOLUTION -AJC
    // vector for storing the analytic solution
    m_var[VL::CPU_ANALYTIC].create_vector(
        m_uiMesh, ot::DVEC_TYPE::OCT_SHARED_NODES, ot::DVEC_LOC::HOST,
        SOLVER_NUM_VARS, true);
    m_var[VL::CPU_ANALYTIC_DIFF].create_vector(
        m_uiMesh, ot::DVEC_TYPE::OCT_SHARED_NODES, ot::DVEC_LOC::HOST,
        SOLVER_NUM_VARS, true);

// re-enable this if you want to solve analytical on a block-wise basis,
// shouldn't be necessary though
#if 0
    // unzipped versions because I think these are easier to use?
    m_var[VL::CPU_ANALYTIC_UZ_IN].create_vector(
        m_uiMesh, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::HOST,
        SOLVER_NUM_VARS, true);
    m_var[VL::CPU_ANALYTIC_DIFF_UZ_IN].create_vector(
        m_uiMesh, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::HOST,
        SOLVER_NUM_VARS, true);
#endif

    m_uiTinfo._m_uiStep = 0;
    m_uiTinfo._m_uiT = 0;
    m_uiTinfo._m_uiTb = dsolve::SOLVER_RK_TIME_BEGIN;
    m_uiTinfo._m_uiTe = dsolve::SOLVER_RK_TIME_END;
    m_uiTinfo._m_uiTh = dsolve::SOLVER_RK45_TIME_STEP_SIZE;

    m_uiElementOrder = dsolve::SOLVER_ELE_ORDER;

    m_uiMinPt = Point(dsolve::SOLVER_GRID_MIN_X, dsolve::SOLVER_GRID_MIN_Y,
                      dsolve::SOLVER_GRID_MIN_Z);
    m_uiMaxPt = Point(dsolve::SOLVER_GRID_MAX_X, dsolve::SOLVER_GRID_MAX_Y,
                      dsolve::SOLVER_GRID_MAX_Z);

    deallocate_deriv_workspace();
    allocate_deriv_workspace(m_uiMesh, 1);

    // deallocate MPI ctx and allocate MPI ctx
    ot::dealloc_mpi_ctx<DendroScalar>(m_uiMesh, m_mpi_ctx, SOLVER_NUM_VARS,
                                      SOLVER_ASYNC_COMM_K);
    ot::alloc_mpi_ctx<DendroScalar>(m_uiMesh, m_mpi_ctx, SOLVER_NUM_VARS,
                                    SOLVER_ASYNC_COMM_K);

    // then initialize the CFD stuff

#ifdef EM4_ENABLE_COMPACT_DERIVS
    dendro_cfd::cfd.set_filter_type(dsolve::SOLVER_FILTER_TYPE);
    dendro_cfd::cfd.set_deriv_type(dsolve::SOLVER_DERIV_TYPE);
    dendro_cfd::cfd.set_second_deriv_type(dsolve::SOLVER_2ND_DERIV_TYPE);
    dendro_cfd::cfd.set_padding_size(dsolve::SOLVER_PADDING_WIDTH);
    dendro_cfd::cfd.set_kim_params(dsolve::SOLVER_KIM_FILTER_KC,
                                   dsolve::SOLVER_KIM_FILTER_EPS);
    dendro_cfd::cfd.set_deriv_boundary_type(dsolve::SOLVER_DERIV_CLOSURE_TYPE);
    dendro_cfd::set_bl_matnum(dsolve::SOLVER_BL_DERIV_MAT_NUM);

    // NOTE: calling this function will reinitialize the entire cfd object.
    // It should calculate everything!
    dendro_cfd::cfd.change_dim_size(2 * dsolve::SOLVER_ELE_ORDER + 1);
    // std::cout << "Reinitialized cfd object with size "
    //           << 2 * dsolve::SOLVER_ELE_ORDER + 1 << std::endl;

#endif

    // set up the appropriate derivs
    dendro_derivs::set_appropriate_derivs(dsolve::SOLVER_PADDING_WIDTH);

    return;
}

SOLVERCtx::~SOLVERCtx() {
    for (unsigned int i = 0; i < VL::END; i++) m_var[i].destroy_vector();

    deallocate_deriv_workspace();
    ot::dealloc_mpi_ctx<DendroScalar>(m_uiMesh, m_mpi_ctx, SOLVER_NUM_VARS,
                                      SOLVER_ASYNC_COMM_K);
}

int SOLVERCtx::rhs(DVec *in, DVec *out, unsigned int sz, DendroScalar time) {
    // all the variables should be packed together.
    // assert(sz == 1);
    // DendroScalar **sVar;
    // in[0].Get2DArray(sVar, false);

    dsolve::timer::t_unzip_async.start();
    this->unzip(*in, m_var[VL::CPU_EV_UZ_IN], dsolve::SOLVER_ASYNC_COMM_K);
    dsolve::timer::t_unzip_async.stop();

#ifdef __PROFILE_CTX__
    this->m_uiCtxpt[ts::CTXPROFILE::RHS].start();
#endif

    DendroScalar *unzipIn[SOLVER_NUM_VARS];
    DendroScalar *unzipOut[SOLVER_NUM_VARS];

    m_var[CPU_EV_UZ_IN].to_2d(unzipIn);
    m_var[CPU_EV_UZ_OUT].to_2d(unzipOut);

    const ot::Block *blkList = m_uiMesh->getLocalBlockList().data();
    const unsigned int numBlocks = m_uiMesh->getLocalBlockList().size();

    solverRHS(unzipOut, unzipIn, blkList, numBlocks);

#ifdef __PROFILE_CTX__
    this->m_uiCtxpt[ts::CTXPROFILE::RHS].stop();
#endif

    // NOTE: here is where dumping to binary file would be appropriate for
    // training/validating

    dsolve::timer::t_zip.start();
    this->zip(m_var[CPU_EV_UZ_OUT], *out);
    dsolve::timer::t_zip.stop();

    return 0;
}

void SOLVERCtx::compute_constraints() {
    // early exit if the constraints have already been computed
    // this is fine for any and all processes, even inactive ones
    if (m_constraintsComputed) return;

    if (m_uiMesh->isActive()) {
        DVec &m_evar = m_var[VL::CPU_EV];
        DVec &m_evar_unz = m_var[VL::CPU_EV_UZ_IN];
        DVec &m_cvar = m_var[VL::CPU_CV];
        DVec &m_cvar_unz = m_var[VL::CPU_CV_UZ_IN];
        this->unzip(m_evar, m_evar_unz, SOLVER_ASYNC_COMM_K);

        DendroScalar *consUnzipVar[dsolve::SOLVER_CONSTRAINT_NUM_VARS];
        DendroScalar *consVar[dsolve::SOLVER_CONSTRAINT_NUM_VARS];

        DendroScalar *evolUnzipVar[dsolve::SOLVER_NUM_VARS];
        DendroScalar *evolVar[dsolve::SOLVER_NUM_VARS];

        m_evar_unz.to_2d(evolUnzipVar);
        m_cvar_unz.to_2d(consUnzipVar);

        m_evar.to_2d(evolVar);
        m_cvar.to_2d(consVar);

        const std::vector<ot::Block> blkList = m_uiMesh->getLocalBlockList();

        unsigned int offset;
        double ptmin[3], ptmax[3];
        unsigned int sz[3];
        unsigned int bflag;
        double dx, dy, dz;
        const Point pt_min(dsolve::SOLVER_COMPD_MIN[0],
                           dsolve::SOLVER_COMPD_MIN[1],
                           dsolve::SOLVER_COMPD_MIN[2]);
        const Point pt_max(dsolve::SOLVER_COMPD_MAX[0],
                           dsolve::SOLVER_COMPD_MAX[1],
                           dsolve::SOLVER_COMPD_MAX[2]);
        const unsigned int PW = dsolve::SOLVER_PADDING_WIDTH;

        for (unsigned int blk = 0; blk < blkList.size(); blk++) {
            offset = blkList[blk].getOffset();
            sz[0] = blkList[blk].getAllocationSzX();
            sz[1] = blkList[blk].getAllocationSzY();
            sz[2] = blkList[blk].getAllocationSzZ();

            bflag = blkList[blk].getBlkNodeFlag();

            dx = blkList[blk].computeDx(pt_min, pt_max);
            dy = blkList[blk].computeDy(pt_min, pt_max);
            dz = blkList[blk].computeDz(pt_min, pt_max);

            ptmin[0] = GRIDX_TO_X(blkList[blk].getBlockNode().minX()) - PW * dx;
            ptmin[1] = GRIDY_TO_Y(blkList[blk].getBlockNode().minY()) - PW * dy;
            ptmin[2] = GRIDZ_TO_Z(blkList[blk].getBlockNode().minZ()) - PW * dz;

            ptmax[0] = GRIDX_TO_X(blkList[blk].getBlockNode().maxX()) + PW * dx;
            ptmax[1] = GRIDY_TO_Y(blkList[blk].getBlockNode().maxY()) + PW * dy;
            ptmax[2] = GRIDZ_TO_Z(blkList[blk].getBlockNode().maxZ()) + PW * dz;

#ifdef EM4_ENABLE_COMPACT_DERIVS
            physical_constraints_compact_derivs(
                consUnzipVar, evolUnzipVar, offset, ptmin, ptmax, sz, bflag);
#else
            physical_constraints(consUnzipVar,
                                 (const DendroScalar **)evolUnzipVar, offset,
                                 ptmin, ptmax, sz, bflag);
#endif
        }

        // end by zipping it back up and then syncing the constraint grid
        this->zip(m_cvar_unz, m_cvar);
        m_uiMesh->readFromGhostBegin(m_cvar.get_vec_ptr(), m_cvar.get_dof());
        m_uiMesh->readFromGhostEnd(m_cvar.get_vec_ptr(), m_cvar.get_dof());

        if (!(m_uiMesh->getMPIRank())) {
            std::cout << "\t\tConstraint computation successfully finished."
                      << std::endl;
        }
    }
    // all meshes will store computed by the time this finishes
    m_constraintsComputed = true;
}

void SOLVERCtx::compute_analytical() {
    // early exit if the analytical has already been computed
    // this is fine for any and all mesh, even inactive ones
    if (m_analyticalComputed) return;

#if 0
    if (m_uiMesh->isActive()) {
        DVec &m_evar = m_var[VL::CPU_EV];
        DVec &m_evar_unz = m_var[VL::CPU_EV_UZ_IN];

        DVec &m_analytic = m_var[VL::CPU_ANALYTIC];
        DVec &m_analytic_unz = m_var[VL::CPU_ANALYTIC_UZ_IN];

        DVec &m_analytic_diff = m_var[VL::CPU_ANALYTIC_DIFF];
        DVec &m_analytic_diff_unz = m_var[VL::CPU_ANALYTIC_DIFF_UZ_IN];

        this->unzip(m_evar, m_evar_unz, SOLVER_ASYNC_COMM_K);

        DendroScalar *evolUnzipVar[dsolve::SOLVER_NUM_VARS];
        DendroScalar *evolVar[dsolve::SOLVER_NUM_VARS];

        DendroScalar *analyticUnzipVar[dsolve::SOLVER_NUM_VARS];
        DendroScalar *analyticVar[dsolve::SOLVER_NUM_VARS];

        DendroScalar *analyticDiffUnzipVar[dsolve::SOLVER_NUM_VARS];
        DendroScalar *analyticDiffVar[dsolve::SOLVER_NUM_VARS];

        m_evar_unz.to_2d(evolUnzipVar);
        m_analytic_unz.to_2d(analyticUnzipVar);
        m_analytic_diff_unz.to_2d(analyticDiffUnzipVar);

        m_evar.to_2d(evolVar);
        m_analytic.to_2d(analyticVar);
        m_analytic_diff.to_2d(analyticDiffVar);

        std::cout << "Now getting local block list and setting up for "
                     "analytical solve"
                  << std::endl;

        const std::vector<ot::Block> blkList = m_uiMesh->getLocalBlockList();

        unsigned int offset;
        double ptmin[3], ptmax[3];
        unsigned int sz[3];
        unsigned int bflag;
        double dx, dy, dz;
        const Point pt_min(dsolve::SOLVER_COMPD_MIN[0],
                           dsolve::SOLVER_COMPD_MIN[1],
                           dsolve::SOLVER_COMPD_MIN[2]);
        const Point pt_max(dsolve::SOLVER_COMPD_MAX[0],
                           dsolve::SOLVER_COMPD_MAX[1],
                           dsolve::SOLVER_COMPD_MAX[2]);
        const unsigned int PW = dsolve::SOLVER_PADDING_WIDTH;

        for (unsigned int blk = 0; blk < blkList.size(); blk++) {
            offset = blkList[blk].getOffset();
            sz[0] = blkList[blk].getAllocationSzX();
            sz[1] = blkList[blk].getAllocationSzY();
            sz[2] = blkList[blk].getAllocationSzZ();

            bflag = blkList[blk].getBlkNodeFlag();

            dx = blkList[blk].computeDx(pt_min, pt_max);
            dy = blkList[blk].computeDy(pt_min, pt_max);
            dz = blkList[blk].computeDz(pt_min, pt_max);

            ptmin[0] = GRIDX_TO_X(blkList[blk].getBlockNode().minX()) - PW * dx;
            ptmin[1] = GRIDY_TO_Y(blkList[blk].getBlockNode().minY()) - PW * dy;
            ptmin[2] = GRIDZ_TO_Z(blkList[blk].getBlockNode().minZ()) - PW * dz;

            ptmax[0] = GRIDX_TO_X(blkList[blk].getBlockNode().maxX()) + PW * dx;
            ptmax[1] = GRIDY_TO_Y(blkList[blk].getBlockNode().maxY()) + PW * dy;
            ptmax[2] = GRIDZ_TO_Z(blkList[blk].getBlockNode().maxZ()) + PW * dz;

            dsolve::analyticalSolEM4_BLOCK(
                analyticUnzipVar, analyticDiffUnzipVar,
                (const DendroScalar **)evolUnzipVar, m_uiTinfo._m_uiT, offset,
                ptmin, ptmax, sz, bflag);
        }

        // now zip things back up, which should help?
        this->zip(m_analytic_unz, m_analytic);
        this->zip(m_analytic_diff_unz, m_analytic_diff);

        // NOTE: not sure if I need to actually do the communication here or not
        // I think BSSN did it simply because of the extraction step for grav
        // waves
        m_uiMesh->readFromGhostBegin(m_analytic.get_vec_ptr(),
                                     m_analytic.get_dof());
        m_uiMesh->readFromGhostEnd(m_analytic.get_vec_ptr(),
                                   m_analytic.get_dof());
        m_uiMesh->readFromGhostBegin(m_analytic_diff.get_vec_ptr(),
                                     m_analytic_diff.get_dof());
        m_uiMesh->readFromGhostEnd(m_analytic_diff.get_vec_ptr(),
                                   m_analytic_diff.get_dof());

        if (!(m_uiMesh->getMPIRank())) {
            std::cout << "\t\tFinished computation of analytical!" << std::endl;
        }
    }
#endif

    if (m_uiMesh->isActive()) {
        DVec &m_evar = m_var[VL::CPU_EV];
        DVec &m_analytic = m_var[VL::CPU_ANALYTIC];
        DVec &m_analytic_diff = m_var[VL::CPU_ANALYTIC_DIFF];

        const ot::TreeNode *pNodes = &(*(m_uiMesh->getAllElements().begin()));
        const unsigned int eleOrder = m_uiMesh->getElementOrder();
        const unsigned int *e2n_cg = &(*(m_uiMesh->getE2NMapping().begin()));
        const unsigned int *e2n_dg = &(*(m_uiMesh->getE2NMapping_DG().begin()));
        const unsigned int nPe = m_uiMesh->getNumNodesPerElement();
        const unsigned int nodeLocalBegin = m_uiMesh->getNodeLocalBegin();
        const unsigned int nodeLocalEnd = m_uiMesh->getNodeLocalEnd();

        DendroScalar *analytical_var[SOLVER_NUM_VARS];
        DendroScalar *analytical_diff[SOLVER_NUM_VARS];
        DendroScalar *zipped_vars[SOLVER_NUM_VARS];

        m_analytic.to_2d(analytical_var);
        m_analytic_diff.to_2d(analytical_diff);
        m_evar.to_2d(zipped_vars);

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
                            m_uiMesh->dg2eijk(nodeLookUp_DG, ownerID, ii_x,
                                              jj_y, kk_z);

                            const DendroScalar len =
                                (double)(1u << (m_uiMaxDepth -
                                                pNodes[ownerID].getLevel()));
                            const DendroScalar x = pNodes[ownerID].getX() +
                                                   ii_x * (len / (eleOrder));
                            const DendroScalar y = pNodes[ownerID].getY() +
                                                   jj_y * (len / (eleOrder));
                            const DendroScalar z = pNodes[ownerID].getZ() +
                                                   kk_z * (len / (eleOrder));

                            dsolve::analyticalSolEM4((double)x, (double)y,
                                                     (double)z,
                                                     m_uiTinfo._m_uiT, var);

                            for (unsigned int v = 0;
                                 v < dsolve::SOLVER_NUM_VARS; v++) {
                                analytical_var[v][nodeLookUp_CG] = var[v];
                                analytical_diff[v][nodeLookUp_CG] =
                                    zipped_vars[v][nodeLookUp_CG] - var[v];
                            }
                        }
                    }
        }

        // NOTE: not sure if I need to actually do the communication here or
        // not I think BSSN did it simply because of the extraction step for
        // grav waves
        m_uiMesh->readFromGhostBegin(m_analytic.get_vec_ptr(),
                                     m_analytic.get_dof());
        m_uiMesh->readFromGhostEnd(m_analytic.get_vec_ptr(),
                                   m_analytic.get_dof());
        m_uiMesh->readFromGhostBegin(m_analytic_diff.get_vec_ptr(),
                                     m_analytic_diff.get_dof());
        m_uiMesh->readFromGhostEnd(m_analytic_diff.get_vec_ptr(),
                                   m_analytic_diff.get_dof());

        if (!(m_uiMesh->getMPIRank())) {
            std::cout << "\t\tFinished computation of analytical!" << std::endl;
        }
    }
    // all meshes will store computed by the time this finishes
    m_analyticalComputed = true;
}

// NOTE: the blkwise computations are *not* considered anymore, but there are a
// few changes made to BSSN code
#if 0
int SOLVERCtx::rhs_blkwise(DVec in, DVec out, const unsigned int *const blkIDs,
                         unsigned int numIds, DendroScalar *blk_time) const {
    DendroScalar **unzipIn;
    DendroScalar **unzipOut;

    assert(in.GetDof() == out.GetDof());
    assert(in.IsUnzip() == out.IsUnzip());

    in.Get2DArray(unzipIn, false);
    out.Get2DArray(unzipOut, false);

    const ot::Block *blkList = m_uiMesh->getLocalBlockList().data();
    const unsigned int numBlocks = m_uiMesh->getLocalBlockList().size();

    unsigned int offset;
    double ptmin[3], ptmax[3];
    unsigned int sz[3];
    unsigned int bflag;
    double dx, dy, dz;
    const Point pt_min(dsolve::SOLVER_COMPD_MIN[0], dsolve::SOLVER_COMPD_MIN[1],
                       dsolve::SOLVER_COMPD_MIN[2]);
    const Point pt_max(dsolve::SOLVER_COMPD_MAX[0], dsolve::SOLVER_COMPD_MAX[1],
                       dsolve::SOLVER_COMPD_MAX[2]);
    const unsigned int PW = dsolve::SOLVER_PADDING_WIDTH;

    for (unsigned int i = 0; i < numIds; i++) {
        const unsigned int blk = blkIDs[i];
        assert(blk < numBlocks);

        offset = blkList[blk].getOffset();
        sz[0] = blkList[blk].getAllocationSzX();
        sz[1] = blkList[blk].getAllocationSzY();
        sz[2] = blkList[blk].getAllocationSzZ();

        bflag = blkList[blk].getBlkNodeFlag();

        dx = blkList[blk].computeDx(pt_min, pt_max);
        dy = blkList[blk].computeDy(pt_min, pt_max);
        dz = blkList[blk].computeDz(pt_min, pt_max);

        ptmin[0] = GRIDX_TO_X(blkList[blk].getBlockNode().minX()) - PW * dx;
        ptmin[1] = GRIDY_TO_Y(blkList[blk].getBlockNode().minY()) - PW * dy;
        ptmin[2] = GRIDZ_TO_Z(blkList[blk].getBlockNode().minZ()) - PW * dz;

        ptmax[0] = GRIDX_TO_X(blkList[blk].getBlockNode().maxX()) + PW * dx;
        ptmax[1] = GRIDY_TO_Y(blkList[blk].getBlockNode().maxY()) + PW * dy;
        ptmax[2] = GRIDZ_TO_Z(blkList[blk].getBlockNode().maxZ()) + PW * dz;

#ifdef SOLVER_RHS_STAGED_COMP
        solverrhs_sep(unzipOut, (const double **)unzipIn, offset, ptmin, ptmax,
                    sz, bflag);
#else
        solverrhs(unzipOut, (const double **)unzipIn, offset, ptmin, ptmax, sz,
                bflag);
#endif
    }

    delete[] unzipIn;
    delete[] unzipOut;

    return 0;
}

int SOLVERCtx::rhs_blk(const DendroScalar *in, DendroScalar *out,
                     unsigned int dof, unsigned int local_blk_id,
                     DendroScalar blk_time) const {
    // return 0;
    // std::cout<<"solver_rhs"<<std::endl;
    DendroScalar **unzipIn = new DendroScalar *[dof];
    DendroScalar **unzipOut = new DendroScalar *[dof];

    const unsigned int blk = local_blk_id;

    const ot::Block *blkList = m_uiMesh->getLocalBlockList().data();
    const unsigned int numBlocks = m_uiMesh->getLocalBlockList().size();
    double ptmin[3], ptmax[3];
    unsigned int sz[3];
    unsigned int bflag;
    double dx, dy, dz;
    const Point pt_min(dsolve::SOLVER_COMPD_MIN[0], dsolve::SOLVER_COMPD_MIN[1],
                       dsolve::SOLVER_COMPD_MIN[2]);
    const Point pt_max(dsolve::SOLVER_COMPD_MAX[0], dsolve::SOLVER_COMPD_MAX[1],
                       dsolve::SOLVER_COMPD_MAX[2]);
    const unsigned int PW = dsolve::SOLVER_PADDING_WIDTH;

    sz[0] = blkList[blk].getAllocationSzX();
    sz[1] = blkList[blk].getAllocationSzY();
    sz[2] = blkList[blk].getAllocationSzZ();

    const unsigned int NN = sz[0] * sz[1] * sz[2];

    for (unsigned int v = 0; v < dof; v++) {
        unzipIn[v] = (DendroScalar *)(in + v * NN);
        unzipOut[v] = (DendroScalar *)(out + v * NN);
    }

    bflag = blkList[blk].getBlkNodeFlag();
    const unsigned int pw = blkList[blk].get1DPadWidth();

    // if(!bflag)
    // {
    //     // for(unsigned int node=0; node < NN; node++)
    //     //     enforce_system_constraints(unzipIn, node);
    //     for(unsigned int k=pw; k < sz[2]-pw; k++)
    //     for(unsigned int j=pw; j < sz[1]-pw; j++)
    //     for(unsigned int i=pw; i < sz[0]-pw; i++)
    //     {
    //         const unsigned nid = k*sz[1]*sz[0] + j*sz[0] + i;
    //         enforce_system_constraints(unzipIn,nid);
    //     }

    // }else
    // {
    //     // note that we can apply enforce solver constraints in the right padd,
    //     at the left boundary block,
    //     // currently we only apply internal parts of the boundary blocks.
    //     for(unsigned int k=pw; k < sz[2]-pw; k++)
    //     for(unsigned int j=pw; j < sz[1]-pw; j++)
    //     for(unsigned int i=pw; i < sz[0]-pw; i++)
    //     {
    //         const unsigned nid = k*sz[1]*sz[0] + j*sz[0] + i;
    //         enforce_system_constraints(unzipIn,nid);
    //     }

    // }

    dx = blkList[blk].computeDx(pt_min, pt_max);
    dy = blkList[blk].computeDy(pt_min, pt_max);
    dz = blkList[blk].computeDz(pt_min, pt_max);

    ptmin[0] = GRIDX_TO_X(blkList[blk].getBlockNode().minX()) - PW * dx;
    ptmin[1] = GRIDY_TO_Y(blkList[blk].getBlockNode().minY()) - PW * dy;
    ptmin[2] = GRIDZ_TO_Z(blkList[blk].getBlockNode().minZ()) - PW * dz;

    ptmax[0] = GRIDX_TO_X(blkList[blk].getBlockNode().maxX()) + PW * dx;
    ptmax[1] = GRIDY_TO_Y(blkList[blk].getBlockNode().maxY()) + PW * dy;
    ptmax[2] = GRIDZ_TO_Z(blkList[blk].getBlockNode().maxZ()) + PW * dz;

#ifdef SOLVER_RHS_STAGED_COMP
    solverrhs_sep(unzipOut, (const DendroScalar **)unzipIn, 0, ptmin, ptmax, sz,
                bflag);
#else
    solverrhs(unzipOut, (const DendroScalar **)unzipIn, 0, ptmin, ptmax, sz,
            bflag);
#endif

    delete[] unzipIn;
    delete[] unzipOut;

    return 0;
}

int SOLVERCtx::pre_stage_blk(DendroScalar *in, unsigned int dof,
                           unsigned int local_blk_id,
                           DendroScalar blk_time) const {
    DendroScalar **unzipIn = new DendroScalar *[dof];
    const unsigned int blk = local_blk_id;

    const ot::Block *blkList = m_uiMesh->getLocalBlockList().data();
    const unsigned int numBlocks = m_uiMesh->getLocalBlockList().size();
    double ptmin[3], ptmax[3];
    unsigned int sz[3];
    unsigned int bflag;
    double dx, dy, dz;
    const Point pt_min(dsolve::SOLVER_COMPD_MIN[0], dsolve::SOLVER_COMPD_MIN[1],
                       dsolve::SOLVER_COMPD_MIN[2]);
    const Point pt_max(dsolve::SOLVER_COMPD_MAX[0], dsolve::SOLVER_COMPD_MAX[1],
                       dsolve::SOLVER_COMPD_MAX[2]);

    sz[0] = blkList[blk].getAllocationSzX();
    sz[1] = blkList[blk].getAllocationSzY();
    sz[2] = blkList[blk].getAllocationSzZ();

    const unsigned int NN = sz[0] * sz[1] * sz[2];

    for (unsigned int v = 0; v < dof; v++) {
        unzipIn[v] = (DendroScalar *)(in + v * NN);
    }

    bflag = blkList[blk].getBlkNodeFlag();
    const unsigned int pw = blkList[blk].get1DPadWidth();

    if (!bflag) {
        for (unsigned int node = 0; node < NN; node++)
            enforce_system_constraints(unzipIn, node);
        // for(unsigned int k=pw; k < sz[2]-pw; k++)
        // for(unsigned int j=pw; j < sz[1]-pw; j++)
        // for(unsigned int i=pw; i < sz[0]-pw; i++)
        // {
        //     const unsigned nid = k*sz[1]*sz[0] + j*sz[0] + i;
        //     enforce_system_constraints(unzipIn,nid);
        // }
    } else {
        // note that we can apply enforce solver constraints in the right padd, at
        // the left boundary block, currently we only apply internal parts of
        // the boundary blocks.
        for (unsigned int k = pw; k < sz[2] - pw; k++)
            for (unsigned int j = pw; j < sz[1] - pw; j++)
                for (unsigned int i = pw; i < sz[0] - pw; i++) {
                    const unsigned nid = k * sz[1] * sz[0] + j * sz[0] + i;
                    enforce_system_constraints(unzipIn, nid);
                }
    }

    delete[] unzipIn;
    return 0;
}

int SOLVERCtx::post_stage_blk(DendroScalar *in, unsigned int dof,
                            unsigned int local_blk_id,
                            DendroScalar blk_time) const {
    return 0;
}

int SOLVERCtx::pre_timestep_blk(DendroScalar *in, unsigned int dof,
                              unsigned int local_blk_id,
                              DendroScalar blk_time) const {
    return 0;
}

int SOLVERCtx::post_timestep_blk(DendroScalar *in, unsigned int dof,
                               unsigned int local_blk_id,
                               DendroScalar blk_time) const {
    DendroScalar **unzipIn = new DendroScalar *[dof];
    const unsigned int blk = local_blk_id;

    const ot::Block *blkList = m_uiMesh->getLocalBlockList().data();
    const unsigned int numBlocks = m_uiMesh->getLocalBlockList().size();
    double ptmin[3], ptmax[3];
    unsigned int sz[3];
    unsigned int bflag;
    double dx, dy, dz;
    const Point pt_min(dsolve::SOLVER_COMPD_MIN[0], dsolve::SOLVER_COMPD_MIN[1],
                       dsolve::SOLVER_COMPD_MIN[2]);
    const Point pt_max(dsolve::SOLVER_COMPD_MAX[0], dsolve::SOLVER_COMPD_MAX[1],
                       dsolve::SOLVER_COMPD_MAX[2]);

    sz[0] = blkList[blk].getAllocationSzX();
    sz[1] = blkList[blk].getAllocationSzY();
    sz[2] = blkList[blk].getAllocationSzZ();

    const unsigned int NN = sz[0] * sz[1] * sz[2];

    for (unsigned int v = 0; v < dof; v++) {
        unzipIn[v] = (DendroScalar *)(in + v * NN);
    }

    bflag = blkList[blk].getBlkNodeFlag();
    const unsigned int pw = blkList[blk].get1DPadWidth();

    if (!bflag) {
        for (unsigned int node = 0; node < NN; node++)
            enforce_system_constraints(unzipIn, node);
        // for(unsigned int k=pw; k < sz[2]-pw; k++)
        // for(unsigned int j=pw; j < sz[1]-pw; j++)
        // for(unsigned int i=pw; i < sz[0]-pw; i++)
        // {
        //     const unsigned nid = k*sz[1]*sz[0] + j*sz[0] + i;
        //     enforce_system_constraints(unzipIn,nid);
        // }
    } else {
        // note that we can apply enforce solver constraints in the right padd, at
        // the left boundary block, currently we only apply internal parts of
        // the boundary blocks.
        for (unsigned int k = pw; k < sz[2] - pw; k++)
            for (unsigned int j = pw; j < sz[1] - pw; j++)
                for (unsigned int i = pw; i < sz[0] - pw; i++) {
                    const unsigned nid = k * sz[1] * sz[0] + j * sz[0] + i;
                    enforce_system_constraints(unzipIn, nid);
                }
    }

    delete[] unzipIn;
    return 0;
}

#endif

int SOLVERCtx::initialize() {
    if (dsolve::SOLVER_RESTORE_SOLVER) {
        this->restore_checkpt();
        return 0;
    }

    if (!dsolve::SOLVER_INIT_GRID_REINITIALIZE_EACH_TIME) this->init_grid();

    bool isRefine = false;
    DendroIntL oldElements, oldElements_g;
    DendroIntL newElements, newElements_g;

    DendroIntL oldGridPoints, oldGridPoints_g;
    DendroIntL newGridPoints, newGridPoints_g;

    unsigned int iterCount = 1;
    const unsigned int max_iter = dsolve::SOLVER_INIT_GRID_ITER;
    const unsigned int rank_global = m_uiMesh->getMPIRankGlobal();
    MPI_Comm gcomm = m_uiMesh->getMPIGlobalCommunicator();

    DendroScalar *unzipVar[dsolve::SOLVER_NUM_VARS];
    unsigned int refineVarIds[dsolve::SOLVER_NUM_REFINE_VARS];

    for (unsigned int vIndex = 0; vIndex < dsolve::SOLVER_NUM_REFINE_VARS;
         vIndex++)
        refineVarIds[vIndex] = dsolve::SOLVER_REFINE_VARIABLE_INDICES[vIndex];

    double wTol = dsolve::SOLVER_WAVELET_TOL;
    std::function<double(double, double, double, double *hx)> waveletTolFunc =
        [](double x, double y, double z, double *hx) {
            return dsolve::computeWTolDCoords(x, y, z, hx);
        };

    DVec &m_evar = m_var[VL::CPU_EV];
    DVec &m_evar_unz = m_var[VL::CPU_EV_UZ_IN];

    do {
        // initialize the grid at each step if desired
        if (dsolve::SOLVER_INIT_GRID_REINITIALIZE_EACH_TIME) this->init_grid();

        this->unzip(m_evar, m_evar_unz, dsolve::SOLVER_ASYNC_COMM_K);
        m_evar_unz.to_2d(unzipVar);
        // isRefine=this->is_remesh();
        // enforce WARM refinement based on refinement initially

        // TODO: don't refine initially if AMR is off
        if (max_iter == 0)
            isRefine = false;
        else {
            if (dsolve::SOLVER_REFINEMENT_MODE ==
                dsolve::RefinementMode::WAMR) {
                // isRefine =
                //     dsolve::isReMeshWAMR(m_uiMesh, (const double
                //     **)unzipVar,
                //                        refineVarIds,
                //                        dsolve::SOLVER_NUM_REFINE_VARS,
                //                        waveletTolFunc,
                //                        dsolve::SOLVER_DENDRO_AMR_FAC);
                isRefine = m_uiMesh->isReMeshUnzip(
                    (const double **)unzipVar, refineVarIds,
                    dsolve::SOLVER_NUM_REFINE_VARS, waveletTolFunc,
                    dsolve::SOLVER_DENDRO_AMR_FAC);

            } else if (dsolve::SOLVER_REFINEMENT_MODE ==
                       dsolve::RefinementMode::REFINE_MODE_NONE) {
                isRefine = false;
            }
        }

        if (isRefine) {
            ot::Mesh *newMesh = this->remesh(dsolve::SOLVER_DENDRO_GRAIN_SZ,
                                             dsolve::SOLVER_LOAD_IMB_TOL,
                                             dsolve::SOLVER_SPLIT_FIX);

            oldElements = m_uiMesh->getNumLocalMeshElements();
            newElements = newMesh->getNumLocalMeshElements();

            oldGridPoints = m_uiMesh->getNumLocalMeshNodes();
            newGridPoints = newMesh->getNumLocalMeshNodes();

            par::Mpi_Allreduce(&oldElements, &oldElements_g, 1, MPI_SUM, gcomm);
            par::Mpi_Allreduce(&newElements, &newElements_g, 1, MPI_SUM, gcomm);

            par::Mpi_Allreduce(&oldGridPoints, &oldGridPoints_g, 1, MPI_SUM,
                               m_uiMesh->getMPIGlobalCommunicator());
            par::Mpi_Allreduce(&newGridPoints, &newGridPoints_g, 1, MPI_SUM,
                               m_uiMesh->getMPIGlobalCommunicator());

            if (!rank_global) {
                std::cout << "[solverCtx] iter : " << iterCount
                          << " (Remesh triggered) ->  old mesh : "
                          << oldElements_g << " new mesh : " << newElements_g
                          << std::endl;

                std::cout << "[solverCtx] iter : " << iterCount
                          << " (Remesh triggered) ->  old mesh (zip nodes) : "
                          << oldGridPoints_g
                          << " new mesh (zip nodes) : " << newGridPoints_g
                          << std::endl;
            }

            this->grid_transfer(newMesh);

            std::swap(m_uiMesh, newMesh);
            delete newMesh;

#ifdef __CUDACC__
// TODO: CUDA STUFF
#endif
        }

        iterCount += 1;

    } while (isRefine &&
             (newElements_g != oldElements_g ||
              newGridPoints_g != oldGridPoints_g) &&
             (iterCount < max_iter));

    // initialize the grid!
    this->init_grid();

    // with the grid now defined, we can allocate the workspace for
    // derivatives
    deallocate_deriv_workspace();
    allocate_deriv_workspace(m_uiMesh, 1);

    // Now we need to make sure we sync the grid because we might have
    // increased our value
    m_uiMesh->readFromGhostBegin(m_var[VL::CPU_EV].get_vec_ptr(),
                                 m_var[VL::CPU_EV].get_dof());
    m_uiMesh->readFromGhostEnd(m_var[VL::CPU_EV].get_vec_ptr(),
                               m_var[VL::CPU_EV].get_dof());

    unsigned int lmin, lmax;
    m_uiMesh->computeMinMaxLevel(lmin, lmax);
    dsolve::SOLVER_RK45_TIME_STEP_SIZE =
        dsolve::SOLVER_CFL_FACTOR *
        ((dsolve::SOLVER_COMPD_MAX[0] - dsolve::SOLVER_COMPD_MIN[0]) *
         ((1u << (m_uiMaxDepth - lmax)) / ((double)dsolve::SOLVER_ELE_ORDER)) /
         ((double)(1u << (m_uiMaxDepth))));
    m_uiTinfo._m_uiTh = dsolve::SOLVER_RK45_TIME_STEP_SIZE;

    if (!m_uiMesh->getMPIRankGlobal()) {
        const DendroScalar dx_finest =
            ((dsolve::SOLVER_COMPD_MAX[0] - dsolve::SOLVER_COMPD_MIN[0]) *
             ((1u << (m_uiMaxDepth - lmax)) /
              ((double)dsolve::SOLVER_ELE_ORDER)) /
             ((double)(1u << (m_uiMaxDepth))));
        const DendroScalar dt_finest = dsolve::SOLVER_CFL_FACTOR * dx_finest;

        std::cout << "================= Grid Info (After init grid "
                     "convergence):========================================"
                     "======="
                     "========"
                  << std::endl;
        std::cout << "lmin: " << lmin << " lmax:" << lmax << std::endl;
        std::cout << "dx: " << dx_finest << std::endl;
        std::cout << "dt: " << dt_finest << std::endl;
        std::cout << "========================================================="
                     "======================================================"
                  << std::endl;
    }

    return 0;
}

int SOLVERCtx::init_grid() {
    DVec &m_evar = m_var[VL::CPU_EV];
    DVec &m_dptr_evar = m_var[VL::GPU_EV];

    const ot::TreeNode *pNodes = &(*(m_uiMesh->getAllElements().begin()));
    const unsigned int eleOrder = m_uiMesh->getElementOrder();
    const unsigned int *e2n_cg = &(*(m_uiMesh->getE2NMapping().begin()));
    const unsigned int *e2n_dg = &(*(m_uiMesh->getE2NMapping_DG().begin()));
    const unsigned int nPe = m_uiMesh->getNumNodesPerElement();
    const unsigned int nodeLocalBegin = m_uiMesh->getNodeLocalBegin();
    const unsigned int nodeLocalEnd = m_uiMesh->getNodeLocalEnd();

    DendroScalar *zipIn[dsolve::SOLVER_NUM_VARS];
    m_evar.to_2d(zipIn);

    DendroScalar var1[dsolve::SOLVER_NUM_VARS];

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

                        const DendroScalar len =
                            (double)(1u << (m_uiMaxDepth -
                                            pNodes[ownerID].getLevel()));
                        const DendroScalar x =
                            pNodes[ownerID].getX() + ii_x * (len / (eleOrder));
                        const DendroScalar y =
                            pNodes[ownerID].getY() + jj_y * (len / (eleOrder));
                        const DendroScalar z =
                            pNodes[ownerID].getZ() + kk_z * (len / (eleOrder));

                        // the data to physical coords takes in the octree
                        // coords and autoconverts them and returns the data
                        // based on things
                        dsolve::initDataFuncToPhysCoords((double)x, (double)y,
                                                         (double)z, var);

                        for (unsigned int v = 0; v < dsolve::SOLVER_NUM_VARS;
                             v++)
                            zipIn[v][nodeLookUp_CG] = var[v];
                    }
                }
    }

    for (unsigned int node = m_uiMesh->getNodeLocalBegin();
         node < m_uiMesh->getNodeLocalEnd(); node++) {
        enforce_system_constraints(zipIn, node);
    }

    return 0;
}

int SOLVERCtx::finalize() { return 0; }

int SOLVERCtx::write_vtu() {
    if (!m_uiMesh->isActive()) return 0;

    DVec &m_evar = m_var[VL::CPU_EV];
    DVec &m_cvar = m_var[VL::CPU_CV];

    DendroScalar *consVar[dsolve::SOLVER_CONSTRAINT_NUM_VARS];

    DendroScalar *evolVar[dsolve::SOLVER_NUM_VARS];

    m_evar.to_2d(evolVar);
    m_cvar.to_2d(consVar);

#ifdef SOLVER_COMPUTE_CONSTRAINTS

    this->compute_constraints();

#endif

#ifdef EM4_COMPUTE_ANALYTICAL

    DVec &m_analytic = m_var[VL::CPU_ANALYTIC];
    DVec &m_analytic_diff = m_var[VL::CPU_ANALYTIC_DIFF];

    this->compute_analytical();
    DendroScalar *analyticVar[dsolve::SOLVER_NUM_VARS];
    DendroScalar *analyticDiffVar[dsolve::SOLVER_NUM_VARS];

    m_analytic.to_2d(analyticVar);
    m_analytic_diff.to_2d(analyticDiffVar);

#endif
    // end EM4_COMPUTE_ANALYTICAL

#ifdef SOLVER_ENABLE_VTU_OUTPUT

    if ((m_uiTinfo._m_uiStep % dsolve::SOLVER_IO_OUTPUT_FREQ) == 0) {
        std::vector<std::string> pDataNames;
        const unsigned int numConstVars =
            dsolve::SOLVER_NUM_CONST_VARS_VTU_OUTPUT;
        const unsigned int numEvolVars =
            dsolve::SOLVER_NUM_EVOL_VARS_VTU_OUTPUT;

#ifdef EM4_COMPUTE_ANALYTICAL
        const unsigned int totalVTUVars = numConstVars + 3 * numEvolVars;
        double *pData[totalVTUVars];
#else
        const unsigned int totalVTUVars = numConstVars + numEvolVars;
        double *pData[totalVTUVars];
#endif
        // end EM4_COMPUTE_ANALYTICAL

        for (unsigned int i = 0; i < numEvolVars; i++) {
            pDataNames.push_back(std::string(
                dsolve::SOLVER_VAR_NAMES[SOLVER_VTU_OUTPUT_EVOL_INDICES[i]]));
            pData[i] = evolVar[SOLVER_VTU_OUTPUT_EVOL_INDICES[i]];
        }

        for (unsigned int i = 0; i < numConstVars; i++) {
            pDataNames.push_back(
                std::string(dsolve::SOLVER_VAR_CONSTRAINT_NAMES
                                [SOLVER_VTU_OUTPUT_CONST_INDICES[i]]));
            pData[numEvolVars + i] =
                consVar[SOLVER_VTU_OUTPUT_CONST_INDICES[i]];
        }

#ifdef EM4_COMPUTE_ANALYTICAL
        for (unsigned int i = 0; i < numEvolVars; i++) {
            pDataNames.push_back(
                std::string(dsolve::SOLVER_VAR_NAMES
                                [SOLVER_VTU_OUTPUT_EVOL_INDICES[i]]) +
                "_ANLYT");
            pData[numEvolVars + numConstVars + i] =
                analyticVar[SOLVER_VTU_OUTPUT_EVOL_INDICES[i]];
        }

        for (unsigned int i = 0; i < numEvolVars; i++) {
            pDataNames.push_back(
                std::string(dsolve::SOLVER_VAR_NAMES
                                [SOLVER_VTU_OUTPUT_EVOL_INDICES[i]]) +
                "_DIFF");
            pData[numEvolVars + numConstVars + numEvolVars + i] =
                analyticDiffVar[SOLVER_VTU_OUTPUT_EVOL_INDICES[i]];
        }
#endif
        // end EM4_COMPUTE_ANALYTICAL

        std::vector<char *> pDataNames_char;
        pDataNames_char.reserve(pDataNames.size());

        for (unsigned int i = 0; i < pDataNames.size(); i++)
            pDataNames_char.push_back(
                const_cast<char *>(pDataNames[i].c_str()));

        const char *fDataNames[] = {"Time", "Cycle"};
        const double fData[] = {m_uiTinfo._m_uiT, (double)m_uiTinfo._m_uiStep};

        char fPrefix[256];
        sprintf(fPrefix, "%s_%06d", dsolve::SOLVER_VTU_FILE_PREFIX.c_str(),
                m_uiTinfo._m_uiStep);

        if (dsolve::SOLVER_VTU_Z_SLICE_ONLY) {
            unsigned int s_val[3] = {1u << (m_uiMaxDepth - 1),
                                     1u << (m_uiMaxDepth - 1),
                                     1u << (m_uiMaxDepth - 1)};
            unsigned int s_norm[3] = {0, 0, 1};
            io::vtk::mesh2vtu_slice(m_uiMesh, s_val, s_norm, fPrefix, 2,
                                    fDataNames, fData, totalVTUVars,
                                    (const char **)&pDataNames_char[0],
                                    (const double **)pData);
        } else
            io::vtk::mesh2vtuFine(
                m_uiMesh, fPrefix, 2, fDataNames, fData, totalVTUVars,
                (const char **)&pDataNames_char[0], (const double **)pData);
    }

#endif
    // end SOLVER_ENABLE_VTU_OUTPUT

    return 0;
}

int SOLVERCtx::write_checkpt() {
    if (!m_uiMesh->isActive()) return 0;

    unsigned int cpIndex;
    (m_uiTinfo._m_uiStep % (2 * dsolve::SOLVER_CHECKPT_FREQ) == 0)
        ? cpIndex = 0
        : cpIndex = 1;  // to support alternate file writing.

    unsigned int rank = m_uiMesh->getMPIRank();
    unsigned int npes = m_uiMesh->getMPICommSize();

    DendroScalar *eVar[SOLVER_NUM_VARS];
    DVec &m_evar = m_var[VL::CPU_EV];
    m_evar.to_2d(eVar);

    char fName[256];
    const ot::TreeNode *pNodes = &(*(m_uiMesh->getAllElements().begin() +
                                     m_uiMesh->getElementLocalBegin()));
    sprintf(fName, "%s_octree_%d_%d.oct",
            dsolve::SOLVER_CHKPT_FILE_PREFIX.c_str(), cpIndex, rank);
    io::checkpoint::writeOctToFile(fName, pNodes,
                                   m_uiMesh->getNumLocalMeshElements());

    unsigned int numVars = dsolve::SOLVER_NUM_VARS;
    const char **varNames = dsolve::SOLVER_VAR_NAMES;

    sprintf(fName, "%s_%d_%d.var", dsolve::SOLVER_CHKPT_FILE_PREFIX.c_str(),
            cpIndex, rank);
    io::checkpoint::writeVecToFile(fName, m_uiMesh, (const double **)eVar,
                                   dsolve::SOLVER_NUM_VARS);

    if (!rank) {
        sprintf(fName, "%s_step_%d.cp",
                dsolve::SOLVER_CHKPT_FILE_PREFIX.c_str(), cpIndex);
        std::cout << "[SOLVERCtx] \t writing checkpoint file : " << fName
                  << std::endl;
        std::ofstream outfile(fName);
        if (!outfile) {
            std::cout << fName << " file open failed " << std::endl;
            return 0;
        }

        json checkPoint;
        checkPoint["DENDRO_TS_TIME_BEGIN"] = m_uiTinfo._m_uiTb;
        checkPoint["DENDRO_TS_TIME_END"] = m_uiTinfo._m_uiTe;
        checkPoint["DENDRO_TS_ELEMENT_ORDER"] = m_uiElementOrder;

        checkPoint["DENDRO_TS_TIME_CURRENT"] = m_uiTinfo._m_uiT;
        checkPoint["DENDRO_TS_STEP_CURRENT"] = m_uiTinfo._m_uiStep;
        checkPoint["DENDRO_TS_TIME_STEP_SIZE"] = m_uiTinfo._m_uiTh;
        checkPoint["DENDRO_TS_LAST_IO_TIME"] = m_uiTinfo._m_uiT;

        checkPoint["DENDRO_TS_WAVELET_TOLERANCE"] = dsolve::SOLVER_WAVELET_TOL;
        checkPoint["DENDRO_TS_LOAD_IMB_TOLERANCE"] =
            dsolve::SOLVER_LOAD_IMB_TOL;
        checkPoint["DENDRO_TS_NUM_VARS"] =
            numVars;  // number of variables to restore.
        checkPoint["DENDRO_TS_ACTIVE_COMM_SZ"] =
            m_uiMesh->getMPICommSize();  // (note that rank 0 is always active).

        outfile << std::setw(4) << checkPoint << std::endl;
        outfile.close();
    }

    return 0;
}

int SOLVERCtx::restore_checkpt() {
    unsigned int numVars = 0;
    std::vector<ot::TreeNode> octree;
    json checkPoint;

    int rank;
    int npes;
    MPI_Comm comm = m_uiMesh->getMPIGlobalCommunicator();
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);

    unsigned int activeCommSz;

    char fName[256];
    unsigned int restoreStatus = 0;
    unsigned int restoreStatusGlobal =
        0;  // 0 indicates successfully restorable.

    ot::Mesh *newMesh;
    unsigned int restoreStep[2];
    restoreStep[0] = 0;
    restoreStep[1] = 0;

    unsigned int restoreFileIndex = 0;

    for (unsigned int cpIndex = 0; cpIndex < 2; cpIndex++) {
        restoreStatus = 0;

        if (!rank) {
            sprintf(fName, "%s_step_%d.cp",
                    dsolve::SOLVER_CHKPT_FILE_PREFIX.c_str(), cpIndex);
            std::ifstream infile(fName);
            if (!infile) {
                std::cout << fName << " file open failed " << std::endl;
                restoreStatus = 1;
            }

            if (restoreStatus == 0) {
                infile >> checkPoint;
                m_uiTinfo._m_uiTb = checkPoint["DENDRO_TS_TIME_BEGIN"];
                m_uiTinfo._m_uiTe = checkPoint["DENDRO_TS_TIME_END"];
                m_uiTinfo._m_uiT = checkPoint["DENDRO_TS_TIME_CURRENT"];
                m_uiTinfo._m_uiStep = checkPoint["DENDRO_TS_STEP_CURRENT"];
                m_uiTinfo._m_uiTh = checkPoint["DENDRO_TS_TIME_STEP_SIZE"];
                m_uiElementOrder = checkPoint["DENDRO_TS_ELEMENT_ORDER"];

                dsolve::SOLVER_WAVELET_TOL =
                    checkPoint["DENDRO_TS_WAVELET_TOLERANCE"];
                dsolve::SOLVER_LOAD_IMB_TOL =
                    checkPoint["DENDRO_TS_LOAD_IMB_TOLERANCE"];

                numVars = checkPoint["DENDRO_TS_NUM_VARS"];
                activeCommSz = checkPoint["DENDRO_TS_ACTIVE_COMM_SZ"];

                restoreStep[cpIndex] = m_uiTinfo._m_uiStep;
            }
        }
    }

    if (!rank) {
        if (restoreStep[0] < restoreStep[1])
            restoreFileIndex = 1;
        else
            restoreFileIndex = 0;
    }

    par::Mpi_Bcast(&restoreFileIndex, 1, 0, comm);

    restoreStatus = 0;
    octree.clear();
    if (!rank)
        std::cout << "[SOLVERCtx] :  Trying to restore from checkpoint index : "
                  << restoreFileIndex << std::endl;

    if (!rank) {
        sprintf(fName, "%s_step_%d.cp",
                dsolve::SOLVER_CHKPT_FILE_PREFIX.c_str(), restoreFileIndex);
        std::ifstream infile(fName);
        if (!infile) {
            std::cout << fName << " file open failed " << std::endl;
            restoreStatus = 1;
        }

        if (restoreStatus == 0) {
            infile >> checkPoint;
            m_uiTinfo._m_uiTb = checkPoint["DENDRO_TS_TIME_BEGIN"];
            m_uiTinfo._m_uiTe = checkPoint["DENDRO_TS_TIME_END"];
            m_uiTinfo._m_uiT = checkPoint["DENDRO_TS_TIME_CURRENT"];
            m_uiTinfo._m_uiStep = checkPoint["DENDRO_TS_STEP_CURRENT"];
            m_uiTinfo._m_uiTh = checkPoint["DENDRO_TS_TIME_STEP_SIZE"];
            m_uiElementOrder = checkPoint["DENDRO_TS_ELEMENT_ORDER"];

            dsolve::SOLVER_WAVELET_TOL =
                checkPoint["DENDRO_TS_WAVELET_TOLERANCE"];
            dsolve::SOLVER_LOAD_IMB_TOL =
                checkPoint["DENDRO_TS_LOAD_IMB_TOLERANCE"];

            numVars = checkPoint["DENDRO_TS_NUM_VARS"];
            activeCommSz = checkPoint["DENDRO_TS_ACTIVE_COMM_SZ"];

            restoreStep[restoreFileIndex] = m_uiTinfo._m_uiStep;
        }
    }

    par::Mpi_Allreduce(&restoreStatus, &restoreStatusGlobal, 1, MPI_MAX, comm);
    if (restoreStatusGlobal == 1) {
        if (!rank)
            std::cout << "[SOLVERCtx] : Restore step failed, restore file "
                         "corrupted. "
                      << std::endl;
        MPI_Abort(comm, 0);
    }

    MPI_Bcast(&m_uiTinfo, sizeof(ts::TSInfo), MPI_BYTE, 0, comm);
    par::Mpi_Bcast(&dsolve::SOLVER_WAVELET_TOL, 1, 0, comm);
    par::Mpi_Bcast(&dsolve::SOLVER_LOAD_IMB_TOL, 1, 0, comm);

    par::Mpi_Bcast(&numVars, 1, 0, comm);
    par::Mpi_Bcast(&m_uiElementOrder, 1, 0, comm);
    par::Mpi_Bcast(&activeCommSz, 1, 0, comm);

    if (activeCommSz > npes) {
        if (!rank)
            std::cout << "[SOLVERCtx] : checkpoint file written from  a larger "
                         "communicator than the current global comm. (i.e. "
                         "communicator shrinking not allowed in the restore "
                         "step. )"
                      << std::endl;

        MPI_Abort(comm, 0);
    }

    bool isActive = (rank < activeCommSz);

    MPI_Comm newComm;
    par::splitComm2way(isActive, &newComm, comm);

    if (isActive) {
        int activeRank;
        int activeNpes;

        MPI_Comm_rank(newComm, &activeRank);
        MPI_Comm_size(newComm, &activeNpes);
        assert(activeNpes == activeCommSz);

        sprintf(fName, "%s_octree_%d_%d.oct",
                dsolve::SOLVER_CHKPT_FILE_PREFIX.c_str(), restoreFileIndex,
                activeRank);
        restoreStatus = io::checkpoint::readOctFromFile(fName, octree);
        assert(par::test::isUniqueAndSorted(octree, newComm));
    }

    par::Mpi_Allreduce(&restoreStatus, &restoreStatusGlobal, 1, MPI_MAX, comm);
    if (restoreStatusGlobal == 1) {
        if (!rank)
            std::cout
                << "[SOLVERCtx]: octree (*.oct) restore file is corrupted "
                << std::endl;
        MPI_Abort(comm, 0);
    }

    newMesh = new ot::Mesh(octree, 1, m_uiElementOrder, activeCommSz, comm);
    newMesh->setDomainBounds(
        Point(dsolve::SOLVER_GRID_MIN_X, dsolve::SOLVER_GRID_MIN_Y,
              dsolve::SOLVER_GRID_MIN_Z),
        Point(dsolve::SOLVER_GRID_MAX_X, dsolve::SOLVER_GRID_MAX_Y,
              dsolve::SOLVER_GRID_MAX_Z));
    // no need to transfer data only to resize the contex variables.
    // this->grid_transfer(newMesh);
    for (unsigned int i = 0; i < VL::END; i++) m_var[i].destroy_vector();

    m_var[VL::CPU_EV].create_vector(newMesh, ot::DVEC_TYPE::OCT_SHARED_NODES,
                                    ot::DVEC_LOC::HOST, SOLVER_NUM_VARS, true);
    m_var[VL::CPU_EV_UZ_IN].create_vector(
        newMesh, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::HOST,
        SOLVER_NUM_VARS, true);
    m_var[VL::CPU_EV_UZ_OUT].create_vector(
        newMesh, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::HOST,
        SOLVER_NUM_VARS, true);

    m_var[VL::CPU_CV].create_vector(newMesh, ot::DVEC_TYPE::OCT_SHARED_NODES,
                                    ot::DVEC_LOC::HOST,
                                    SOLVER_CONSTRAINT_NUM_VARS, true);
    m_var[VL::CPU_CV_UZ_IN].create_vector(
        newMesh, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::HOST,
        SOLVER_CONSTRAINT_NUM_VARS, true);

    // make sure to reallocate the analytic vector
    m_var[VL::CPU_ANALYTIC].create_vector(
        newMesh, ot::DVEC_TYPE::OCT_SHARED_NODES, ot::DVEC_LOC::HOST,
        SOLVER_NUM_VARS, true);
    m_var[VL::CPU_ANALYTIC_DIFF].create_vector(
        newMesh, ot::DVEC_TYPE::OCT_SHARED_NODES, ot::DVEC_LOC::HOST,
        SOLVER_NUM_VARS, true);

// re-enable this if you want to solve analytical on a block-wise basis,
// shouldn't be necessary though
#if 0
    m_var[VL::CPU_ANALYTIC_UZ_IN].create_vector(
        newMesh, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::HOST,
        SOLVER_NUM_VARS, true);
    m_var[VL::CPU_ANALYTIC_DIFF_UZ_IN].create_vector(
        newMesh, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::HOST,
        SOLVER_NUM_VARS, true);
#endif

    ot::dealloc_mpi_ctx<DendroScalar>(m_uiMesh, m_mpi_ctx, SOLVER_NUM_VARS,
                                      SOLVER_ASYNC_COMM_K);
    ot::alloc_mpi_ctx<DendroScalar>(newMesh, m_mpi_ctx, SOLVER_NUM_VARS,
                                    SOLVER_ASYNC_COMM_K);

    // only reads the evolution variables.
    if (isActive) {
        int activeRank;
        int activeNpes;

        DendroScalar *inVec[SOLVER_NUM_VARS];
        DVec &m_evar = m_var[VL::CPU_EV];
        m_evar.to_2d(inVec);

        MPI_Comm_rank(newComm, &activeRank);
        MPI_Comm_size(newComm, &activeNpes);
        assert(activeNpes == activeCommSz);

        sprintf(fName, "%s_%d_%d.var", dsolve::SOLVER_CHKPT_FILE_PREFIX.c_str(),
                restoreFileIndex, activeRank);
        restoreStatus = io::checkpoint::readVecFromFile(
            fName, newMesh, inVec, dsolve::SOLVER_NUM_VARS);
    }

    MPI_Comm_free(&newComm);
    par::Mpi_Allreduce(&restoreStatus, &restoreStatusGlobal, 1, MPI_MAX, comm);
    if (restoreStatusGlobal == 1) {
        if (!rank)
            std::cout << "[SOLVERCtx]: varible (*.var) restore file currupted "
                      << std::endl;
        MPI_Abort(comm, 0);
    }

    std::swap(m_uiMesh, newMesh);
    delete newMesh;

    // realloc solver deriv space
    deallocate_deriv_workspace();
    allocate_deriv_workspace(m_uiMesh, 1);

    unsigned int localSz = m_uiMesh->getNumLocalMeshElements();
    unsigned int totalElems = 0;
    par::Mpi_Allreduce(&localSz, &totalElems, 1, MPI_SUM, comm);

    if (!rank)
        std::cout << " checkpoint at step : " << m_uiTinfo._m_uiStep
                  << "active Comm. sz: " << activeCommSz
                  << " restore successful: "
                  << " restored mesh size: " << totalElems << std::endl;

    m_uiIsETSSynced = false;
    return 0;
}

int SOLVERCtx::post_stage(DVec &sIn) { return 0; }

int SOLVERCtx::post_timestep(DVec &sIn) {
    // we need to enforce constraint before computing the HAM and MOM_i
    // constraints.
    DendroScalar *evar[SOLVER_NUM_VARS];
    sIn.to_2d(evar);
    for (unsigned int node = m_uiMesh->getNodeLocalBegin();
         node < m_uiMesh->getNodeLocalEnd(); node++)
        enforce_system_constraints(evar, node);

    return 0;
}

int SOLVERCtx::pre_timestep(DVec &sIn) { return 0; }

int SOLVERCtx::pre_stage(DVec &sIn) { return 0; }

bool SOLVERCtx::is_remesh() {
    bool isRefine = false;
    if (dsolve::SOLVER_ENABLE_BLOCK_ADAPTIVITY) return false;

    MPI_Comm comm = m_uiMesh->getMPIGlobalCommunicator();

    DVec &m_evar = m_var[VL::CPU_EV];
    DVec &m_evar_unz = m_var[VL::CPU_EV_UZ_IN];

    this->unzip(m_evar, m_evar_unz, dsolve::SOLVER_ASYNC_COMM_K);

    DendroScalar *unzipVar[SOLVER_NUM_VARS];
    m_evar_unz.to_2d(unzipVar);

    unsigned int refineVarIds[dsolve::SOLVER_NUM_REFINE_VARS];
    for (unsigned int vIndex = 0; vIndex < dsolve::SOLVER_NUM_REFINE_VARS;
         vIndex++)
        refineVarIds[vIndex] = dsolve::SOLVER_REFINE_VARIABLE_INDICES[vIndex];

    double wTol = dsolve::SOLVER_WAVELET_TOL;
    std::function<double(double, double, double, double *hx)> waveletTolFunc =
        [](double x, double y, double z, double *hx) {
            return dsolve::computeWTolDCoords(x, y, z, hx);
        };

    if (dsolve::SOLVER_REFINEMENT_MODE == dsolve::RefinementMode::WAMR) {
        // isRefine =
        //     dsolve::isReMeshWAMR(m_uiMesh, (const double **)unzipVar,
        //                        refineVarIds,
        //                        dsolve::SOLVER_NUM_REFINE_VARS,
        //                        waveletTolFunc,
        //                        dsolve::SOLVER_DENDRO_AMR_FAC);
        isRefine = m_uiMesh->isReMeshUnzip(
            (const double **)unzipVar, refineVarIds,
            dsolve::SOLVER_NUM_REFINE_VARS, waveletTolFunc,
            dsolve::SOLVER_DENDRO_AMR_FAC);
    } else if (dsolve::SOLVER_REFINEMENT_MODE ==
               dsolve::RefinementMode::REFINE_MODE_NONE) {
        isRefine = false;
    }

    return isRefine;
}

DVec &SOLVERCtx::get_evolution_vars() { return m_var[CPU_EV]; }

DVec &SOLVERCtx::get_constraint_vars() { return m_var[CPU_CV]; }

int SOLVERCtx::terminal_output() {
    if (m_uiMesh->isActive()) {
        std::streamsize ss = std::cout.precision();
        std::streamsize sw = std::cout.width();
        DVec &m_evar = m_var[VL::CPU_EV];
        DendroScalar *zippedUp[SOLVER_NUM_VARS];
        m_var[VL::CPU_EV].to_2d(zippedUp);

        // update cout precision and scientific view

        std::cout << std::scientific;

        std::cout.precision(7);

        for (unsigned int i = 0; i < dsolve::SOLVER_NUM_CONSOLE_OUTPUT_VARS;
             i++) {
            unsigned int v = dsolve::SOLVER_CONSOLE_OUTPUT_VARS[i];
            double l_min = vecMin(&zippedUp[v][m_uiMesh->getNodeLocalBegin()],
                                  (m_uiMesh->getNumLocalMeshNodes()),
                                  m_uiMesh->getMPICommunicator());
            double l_max = vecMax(&zippedUp[v][m_uiMesh->getNodeLocalBegin()],
                                  (m_uiMesh->getNumLocalMeshNodes()),
                                  m_uiMesh->getMPICommunicator());
            double l2_norm = normL2(&zippedUp[v][m_uiMesh->getNodeLocalBegin()],
                                    (m_uiMesh->getNumLocalMeshNodes()),
                                    m_uiMesh->getMPICommunicator());

            if (!(m_uiMesh->getMPIRank())) {
                std::cout << "\t[var]:  " << std::setw(12)
                          << SOLVER_VAR_NAMES[v];
                std::cout << " (min, max, l2) : \t ( " << l_min << ", " << l_max
                          << ", " << l2_norm << ") " << std::endl;
            }
        }

        // and then the difference to analytical!

#ifdef EM4_COMPUTE_ANALYTICAL

        this->compute_analytical();

        DendroScalar *zippedUpAnalyticalDiff[SOLVER_NUM_VARS];
        m_var[VL::CPU_ANALYTIC_DIFF].to_2d(zippedUpAnalyticalDiff);

        for (unsigned int i = 0; i < dsolve::SOLVER_NUM_CONSOLE_OUTPUT_VARS;
             i++) {
            unsigned int v = dsolve::SOLVER_CONSOLE_OUTPUT_VARS[i];
            double l_min = vecMin(
                &zippedUpAnalyticalDiff[v][m_uiMesh->getNodeLocalBegin()],
                (m_uiMesh->getNumLocalMeshNodes()),
                m_uiMesh->getMPICommunicator());
            double l_max = vecMax(
                &zippedUpAnalyticalDiff[v][m_uiMesh->getNodeLocalBegin()],
                (m_uiMesh->getNumLocalMeshNodes()),
                m_uiMesh->getMPICommunicator());
            double l2_norm = normL2(
                &zippedUpAnalyticalDiff[v][m_uiMesh->getNodeLocalBegin()],
                (m_uiMesh->getNumLocalMeshNodes()),
                m_uiMesh->getMPICommunicator());

            if (!(m_uiMesh->getMPIRank())) {
                std::cout << "\t[var]:  " << std::setw(12)
                          << std::string(SOLVER_VAR_NAMES[v]) + "_DIFF";
                std::cout << " (min, max, l2) : \t ( " << l_min << ", " << l_max
                          << ", " << l2_norm << ") " << std::endl;
            }
        }

#endif

#ifdef SOLVER_COMPUTE_CONSTRAINTS
        this->compute_constraints();

        DendroScalar *zippedUpConstraints[SOLVER_CONSTRAINT_NUM_VARS];
        m_var[VL::CPU_CV].to_2d(zippedUpConstraints);

        for (unsigned int i = 0;
             i < dsolve::SOLVER_NUM_CONSOLE_OUTPUT_CONSTRAINTS; i++) {
            unsigned int v = dsolve::SOLVER_CONSOLE_OUTPUT_CONSTRAINTS[i];

            double l_min =
                vecMin(&zippedUpConstraints[v][m_uiMesh->getNodeLocalBegin()],
                       (m_uiMesh->getNumLocalMeshNodes()),
                       m_uiMesh->getMPICommunicator());
            double l_max =
                vecMax(&zippedUpConstraints[v][m_uiMesh->getNodeLocalBegin()],
                       (m_uiMesh->getNumLocalMeshNodes()),
                       m_uiMesh->getMPICommunicator());
            double l2_norm =
                normL2(&zippedUpConstraints[v][m_uiMesh->getNodeLocalBegin()],
                       (m_uiMesh->getNumLocalMeshNodes()),
                       m_uiMesh->getMPICommunicator());

            if (!(m_uiMesh->getMPIRank())) {
                std::cout << "\t[const]:" << std::setw(12)
                          << std::string(SOLVER_VAR_CONSTRAINT_NAMES[v]);
                std::cout << " (min, max, l2) : \t ( " << l_min << ", " << l_max
                          << ", " << l2_norm << ") " << std::endl;
            }
        }

#endif

        std::cout.precision(ss);
        std::cout << std::setw(sw);
        std::cout.unsetf(std::ios_base::floatfield);
    }

    return 0;
}

int SOLVERCtx::grid_transfer(const ot::Mesh *m_new) {
#ifdef __PROFILE_CTX__
    m_uiCtxpt[ts::CTXPROFILE::GRID_TRASFER].start();
#endif
    DVec &m_evar = m_var[VL::CPU_EV];
    DVec::grid_transfer(m_uiMesh, m_new, m_evar);
    // printf("igt ended\n");

    m_var[VL::CPU_CV].destroy_vector();
    m_var[VL::CPU_CV_UZ_IN].destroy_vector();

    m_var[VL::CPU_EV_UZ_IN].destroy_vector();
    m_var[VL::CPU_EV_UZ_OUT].destroy_vector();

    m_var[VL::CPU_ANALYTIC].destroy_vector();
    m_var[VL::CPU_ANALYTIC_DIFF].destroy_vector();

    m_var[VL::CPU_CV].create_vector(m_new, ot::DVEC_TYPE::OCT_SHARED_NODES,
                                    ot::DVEC_LOC::HOST,
                                    SOLVER_CONSTRAINT_NUM_VARS, true);
    m_var[VL::CPU_CV_UZ_IN].create_vector(
        m_new, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::HOST,
        SOLVER_CONSTRAINT_NUM_VARS, true);

    m_var[VL::CPU_EV_UZ_IN].create_vector(
        m_new, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::HOST,
        SOLVER_NUM_VARS, true);
    m_var[VL::CPU_EV_UZ_OUT].create_vector(
        m_new, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::HOST,
        SOLVER_NUM_VARS, true);

    // make sure to reallocate the analytic vector
    m_var[VL::CPU_ANALYTIC].create_vector(
        m_new, ot::DVEC_TYPE::OCT_SHARED_NODES, ot::DVEC_LOC::HOST,
        SOLVER_NUM_VARS, true);
    m_var[VL::CPU_ANALYTIC_DIFF].create_vector(
        m_new, ot::DVEC_TYPE::OCT_SHARED_NODES, ot::DVEC_LOC::HOST,
        SOLVER_NUM_VARS, true);

// re-enable this if you want to solve analytical on a block-wise basis,
// shouldn't be necessary though
#if 0
    m_var[VL::CPU_ANALYTIC_UZ_IN].create_vector(
        m_new, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::HOST,
        SOLVER_NUM_VARS, true);
    m_var[VL::CPU_ANALYTIC_DIFF_UZ_IN].create_vector(
        m_new, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::HOST,
        SOLVER_NUM_VARS, true);
#endif

    ot::dealloc_mpi_ctx<DendroScalar>(m_uiMesh, m_mpi_ctx, SOLVER_NUM_VARS,
                                      SOLVER_ASYNC_COMM_K);
    ot::alloc_mpi_ctx<DendroScalar>(m_new, m_mpi_ctx, SOLVER_NUM_VARS,
                                    SOLVER_ASYNC_COMM_K);

    m_uiIsETSSynced = false;

#ifdef __PROFILE_CTX__
    m_uiCtxpt[ts::CTXPROFILE::GRID_TRASFER].stop();
#endif
    return 0;
}

unsigned int SOLVERCtx::compute_lts_ts_offset() {
    const ot::Block *blkList = m_uiMesh->getLocalBlockList().data();
    const unsigned int numBlocks = m_uiMesh->getLocalBlockList().size();
    const ot::TreeNode *pNodes = m_uiMesh->getAllElements().data();

    const unsigned int ldiff = 0;

    unsigned int lmin, lmax;
    m_uiMesh->computeMinMaxLevel(lmin, lmax);
    SOLVER_LTS_TS_OFFSET = 0;

    const double dt_min =
        dsolve::SOLVER_CFL_FACTOR *
        ((dsolve::SOLVER_COMPD_MAX[0] - dsolve::SOLVER_COMPD_MIN[0]) *
         ((1u << (m_uiMaxDepth - lmax)) / ((double)dsolve::SOLVER_ELE_ORDER)) /
         ((double)(1u << (m_uiMaxDepth))));
    const double dt_eta_fac = dt_min * SOLVER_ETA_CONST;

    for (unsigned int blk = 0; blk < numBlocks; blk++) {
        const unsigned int blev =
            pNodes[blkList[blk].getLocalElementBegin()].getLevel();
        ot::TreeNode blkNode = blkList[blk].getBlockNode();
        const unsigned int szby2 =
            (1u << (m_uiMaxDepth - blkNode.getLevel() - 1));

        Point pt_oct(blkNode.minX() + szby2, blkNode.minY() + szby2,
                     blkNode.minZ() + szby2);
        Point pt_domain;
        m_uiMesh->octCoordToDomainCoord(pt_oct, pt_domain);
        const double r_coord = pt_domain.abs();

        double eta;
        // TODO: find out what to do here for SOLVER
        // THIS IS FROM BSSN CODE:
        // if (dsolve::RIT_ETA_FUNCTION == 0) {
        //     // HAD eta function
        //     eta = ETA_CONST;
        //     if (r_coord >= ETA_R0) {
        //         eta *= pow((ETA_R0 / r_coord), ETA_DAMPING_EXP);
        //     }
        // } else {
        //     // RIT eta function
        //     double w = r_coord / dsolve::RIT_ETA_WIDTH;
        //     double arg = -w * w * w * w;
        //     eta = (dsolve::RIT_ETA_CENTRAL - dsolve::RIT_ETA_OUTER) *
        //     exp(arg) +
        //           dsolve::RIT_ETA_OUTER;
        // }
        // so if the function is zero we can just use this
        eta = SOLVER_ETA_CONST;
        if (r_coord >= SOLVER_ETA_R0) {
            eta *= pow((SOLVER_ETA_R0 / r_coord), SOLVER_ETA_DAMPING_EXP);
        }

        const double dt_eta = dt_eta_fac * (1 / eta);
        const double dt_cfl = (1u << (lmax - blev)) * dt_min;

        const double dt_feasible = std::min(dt_eta, dt_cfl);

        if (dt_feasible > dt_min) {
            unsigned int lts_offset =
                lmax - blev - std::floor(std::log2(dt_feasible / dt_min));
            if (SOLVER_LTS_TS_OFFSET < lts_offset)
                SOLVER_LTS_TS_OFFSET = lts_offset;
        }
    }

    unsigned int lts_offset_max = 0;
    par::Mpi_Allreduce(&SOLVER_LTS_TS_OFFSET, &lts_offset_max, 1, MPI_MAX,
                       m_uiMesh->getMPIGlobalCommunicator());
    SOLVER_LTS_TS_OFFSET = lts_offset_max;

    if (m_uiMesh->isActive() && (!(m_uiMesh->getMPIRank())))
        std::cout << "LTS offset : " << SOLVER_LTS_TS_OFFSET << std::endl;

    return SOLVER_LTS_TS_OFFSET;
}

unsigned int SOLVERCtx::getBlkTimestepFac(unsigned int blev, unsigned int lmin,
                                          unsigned int lmax) {
    const unsigned int ldiff = SOLVER_LTS_TS_OFFSET;
    if ((lmax - blev) <= ldiff)
        return 1;
    else {
        return 1u << (lmax - blev - ldiff);
    }
}

}  // end of namespace dsolve.
