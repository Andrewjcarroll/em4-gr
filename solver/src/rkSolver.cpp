//
// Created by David
//

/**
 * @brief contains RK time stepper for SOLVER equations.
 * @author Milinda Fernando
 * School of Computing, University of Utah
 *
 * */

#include "rkSolver.h"

namespace ode {
namespace solver {

RK_SOLVER::RK_SOLVER(ot::Mesh *pMesh, DendroScalar pTBegin, DendroScalar pTEnd,
                     DendroScalar pTh, RKType rkType)
    : RK(pMesh, pTBegin, pTEnd, pTh) {
    m_uiRKType = rkType;

    // allocate memory for the variables.
    m_uiVar = new DendroScalar *[dsolve::SOLVER_NUM_VARS];
    for (unsigned int index = 0; index < dsolve::SOLVER_NUM_VARS; index++)
        m_uiVar[index] = m_uiMesh->createVector<DendroScalar>();

    m_uiPrevVar = new DendroScalar *[dsolve::SOLVER_NUM_VARS];
    for (unsigned int index = 0; index < dsolve::SOLVER_NUM_VARS; index++)
        m_uiPrevVar[index] = m_uiMesh->createVector<DendroScalar>();

    m_uiVarIm = new DendroScalar *[dsolve::SOLVER_NUM_VARS];
    for (unsigned int index = 0; index < dsolve::SOLVER_NUM_VARS; index++)
        m_uiVarIm[index] = m_uiMesh->createVector<DendroScalar>();

    if (m_uiRKType == RKType::RK3)
        m_uiNumRKStages = dsolve::SOLVER_RK3_STAGES;
    else if (m_uiRKType == RKType::RK4)
        m_uiNumRKStages = dsolve::SOLVER_RK4_STAGES;
    else if (m_uiRKType == RKType::RK45)
        m_uiNumRKStages = dsolve::SOLVER_RK45_STAGES;
    else {
        if (!(pMesh->getMPIRankGlobal()))
            std::cout << "[RK Solver Error]: undefined rk solver type"
                      << std::endl;

        exit(0);
    }

    m_uiStage = new DendroScalar **[m_uiNumRKStages];
    for (unsigned int stage = 0; stage < m_uiNumRKStages; stage++) {
        m_uiStage[stage] = new DendroScalar *[dsolve::SOLVER_NUM_VARS];
        for (unsigned int index = 0; index < dsolve::SOLVER_NUM_VARS; index++)
            m_uiStage[stage][index] = m_uiMesh->createVector<DendroScalar>();
    }

    m_uiUnzipVar = new DendroScalar *[dsolve::SOLVER_NUM_VARS];
    for (unsigned int index = 0; index < dsolve::SOLVER_NUM_VARS; index++)
        m_uiUnzipVar[index] = m_uiMesh->createUnZippedVector<DendroScalar>();

    m_uiUnzipVarRHS = new DendroScalar *[dsolve::SOLVER_NUM_VARS];
    for (unsigned int index = 0; index < dsolve::SOLVER_NUM_VARS; index++)
        m_uiUnzipVarRHS[index] = m_uiMesh->createUnZippedVector<DendroScalar>();

    // allocate memory for the constraint variables.
    m_uiConstraintVars = new DendroScalar *[dsolve::SOLVER_CONSTRAINT_NUM_VARS];
    for (unsigned int index = 0; index < dsolve::SOLVER_CONSTRAINT_NUM_VARS;
         index++)
        m_uiConstraintVars[index] = m_uiMesh->createVector<DendroScalar>();

    m_uiUnzipConstraintVars =
        new DendroScalar *[dsolve::SOLVER_CONSTRAINT_NUM_VARS];
    for (unsigned int index = 0; index < dsolve::SOLVER_CONSTRAINT_NUM_VARS;
         index++)
        m_uiUnzipConstraintVars[index] =
            m_uiMesh->createUnZippedVector<DendroScalar>();

    // mpi communication
    m_uiSendNodeBuf = new DendroScalar *[dsolve::SOLVER_ASYNC_COMM_K];
    m_uiRecvNodeBuf = new DendroScalar *[dsolve::SOLVER_ASYNC_COMM_K];

    m_uiSendReqs = new MPI_Request *[dsolve::SOLVER_ASYNC_COMM_K];
    m_uiRecvReqs = new MPI_Request *[dsolve::SOLVER_ASYNC_COMM_K];
    m_uiSendSts = new MPI_Status *[dsolve::SOLVER_ASYNC_COMM_K];
    m_uiRecvSts = new MPI_Status *[dsolve::SOLVER_ASYNC_COMM_K];

    for (unsigned int index = 0; index < dsolve::SOLVER_ASYNC_COMM_K; index++) {
        m_uiSendNodeBuf[index] = NULL;
        m_uiRecvNodeBuf[index] = NULL;

        m_uiSendReqs[index] = NULL;
        m_uiRecvReqs[index] = NULL;
        m_uiSendSts[index] = NULL;
        m_uiRecvSts[index] = NULL;
    }

    if (m_uiMesh->isActive()) {
        // allocate mpi comm. reqs and status
        for (unsigned int index = 0; index < dsolve::SOLVER_ASYNC_COMM_K;
             index++) {
            if (m_uiMesh->getGhostExcgTotalSendNodeCount() != 0)
                m_uiSendNodeBuf[index] =
                    new DendroScalar[m_uiMesh
                                         ->getGhostExcgTotalSendNodeCount()];
            if (m_uiMesh->getGhostExcgTotalRecvNodeCount() != 0)
                m_uiRecvNodeBuf[index] =
                    new DendroScalar[m_uiMesh
                                         ->getGhostExcgTotalRecvNodeCount()];

            if (m_uiMesh->getSendProcListSize() != 0) {
                m_uiSendReqs[index] =
                    new MPI_Request[m_uiMesh->getSendProcListSize()];
                m_uiSendSts[index] =
                    new MPI_Status[m_uiMesh->getSendProcListSize()];
            }

            if (m_uiMesh->getRecvProcListSize() != 0) {
                m_uiRecvReqs[index] =
                    new MPI_Request[m_uiMesh->getRecvProcListSize()];
                m_uiRecvSts[index] =
                    new MPI_Status[m_uiMesh->getRecvProcListSize()];
            }
        }
    }
    // refresh the derivative workspace
    dsolve::deallocate_deriv_workspace();
    dsolve::allocate_deriv_workspace(m_uiMesh, 1);
}

RK_SOLVER::~RK_SOLVER() {
    for (unsigned int index = 0; index < dsolve::SOLVER_NUM_VARS; index++) {
        delete[] m_uiVar[index];
        delete[] m_uiPrevVar[index];
        delete[] m_uiVarIm[index];
        delete[] m_uiUnzipVar[index];
        delete[] m_uiUnzipVarRHS[index];
    }

    delete[] m_uiVar;
    delete[] m_uiPrevVar;
    delete[] m_uiVarIm;
    delete[] m_uiUnzipVar;
    delete[] m_uiUnzipVarRHS;

    for (unsigned int stage = 0; stage < m_uiNumRKStages; stage++)
        for (unsigned int index = 0; index < dsolve::SOLVER_NUM_VARS; index++)
            delete[] m_uiStage[stage][index];

    for (unsigned int stage = 0; stage < m_uiNumRKStages; stage++)
        delete[] m_uiStage[stage];

    delete[] m_uiStage;

    // deallocate memory for the constraint variables.
    for (unsigned int index = 0; index < dsolve::SOLVER_CONSTRAINT_NUM_VARS;
         index++) {
        delete[] m_uiConstraintVars[index];
        delete[] m_uiUnzipConstraintVars[index];
    }

    delete[] m_uiConstraintVars;
    delete[] m_uiUnzipConstraintVars;

    // mpi communication
    for (unsigned int index = 0; index < dsolve::SOLVER_ASYNC_COMM_K; index++) {
        delete[] m_uiSendNodeBuf[index];
        delete[] m_uiRecvNodeBuf[index];

        delete[] m_uiSendReqs[index];
        delete[] m_uiRecvReqs[index];

        delete[] m_uiSendSts[index];
        delete[] m_uiRecvSts[index];
    }

    delete[] m_uiSendNodeBuf;
    delete[] m_uiRecvNodeBuf;

    delete[] m_uiSendReqs;
    delete[] m_uiSendSts;
    delete[] m_uiRecvReqs;
    delete[] m_uiRecvSts;

    dsolve::deallocate_deriv_workspace();
}

void RK_SOLVER::applyInitialConditions(DendroScalar **zipIn) {
    unsigned int nodeLookUp_CG;
    unsigned int nodeLookUp_DG;
    double x, y, z, len;
    const ot::TreeNode *pNodes = &(*(m_uiMesh->getAllElements().begin()));
    unsigned int ownerID, ii_x, jj_y, kk_z;
    unsigned int eleOrder = m_uiMesh->getElementOrder();
    const unsigned int *e2n_cg = &(*(m_uiMesh->getE2NMapping().begin()));
    const unsigned int *e2n_dg = &(*(m_uiMesh->getE2NMapping_DG().begin()));
    const unsigned int nPe = m_uiMesh->getNumNodesPerElement();
    const unsigned int nodeLocalBegin = m_uiMesh->getNodeLocalBegin();
    const unsigned int nodeLocalEnd = m_uiMesh->getNodeLocalEnd();

    double *var = new double[dsolve::SOLVER_NUM_VARS];

    double mp, mm, mp_adm, mm_adm, E, J1, J2, J3;
    // set the TP communicator.
    // COULD FIX THIS?
    // if (dsolve::SOLVER_ID_TYPE == 0) {

    // }

    for (unsigned int elem = m_uiMesh->getElementLocalBegin();
         elem < m_uiMesh->getElementLocalEnd(); elem++) {
        for (unsigned int k = 0; k < (eleOrder + 1); k++)
            for (unsigned int j = 0; j < (eleOrder + 1); j++)
                for (unsigned int i = 0; i < (eleOrder + 1); i++) {
                    nodeLookUp_CG = e2n_cg[elem * nPe +
                                           k * (eleOrder + 1) * (eleOrder + 1) +
                                           j * (eleOrder + 1) + i];
                    if (nodeLookUp_CG >= nodeLocalBegin &&
                        nodeLookUp_CG < nodeLocalEnd) {
                        nodeLookUp_DG =
                            e2n_dg[elem * nPe +
                                   k * (eleOrder + 1) * (eleOrder + 1) +
                                   j * (eleOrder + 1) + i];
                        m_uiMesh->dg2eijk(nodeLookUp_DG, ownerID, ii_x, jj_y,
                                          kk_z);
                        len = (double)(1u << (m_uiMaxDepth -
                                              pNodes[ownerID].getLevel()));
                        x = pNodes[ownerID].getX() + ii_x * (len / (eleOrder));
                        y = pNodes[ownerID].getY() + jj_y * (len / (eleOrder));
                        z = pNodes[ownerID].getZ() + kk_z * (len / (eleOrder));

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
        // if ( node == 101 ) {
        // std::cout << "yo 1 node=" << node << std::endl ;
        //}

        enforce_system_constraints(zipIn, node);
    }

    delete[] var;
}

void RK_SOLVER::initialGridConverge() {
    applyInitialConditions(m_uiPrevVar);

    // isRefine refers to if we should try to refine further
    bool isRefine = false;
    DendroIntL oldElements, oldElements_g;
    DendroIntL newElements, newElements_g;

    DendroIntL oldGridPoints, oldGridPoints_g;
    DendroIntL newGridPoints, newGridPoints_g;

    // refine based on all the variables
    const unsigned int refineNumVars = dsolve::SOLVER_NUM_REFINE_VARS;
    unsigned int refineVarIds[refineNumVars];
    for (unsigned int vIndex = 0; vIndex < refineNumVars; vIndex++)
        refineVarIds[vIndex] = dsolve::SOLVER_REFINE_VARIABLE_INDICES[vIndex];

    double wTol = dsolve::SOLVER_WAVELET_TOL;
    std::function<double(double, double, double, double *)> waveletTolFunc =
        [](double x, double y, double z, double *hx) {
            return dsolve::computeWTolDCoords(x, y, z, hx);
        };
    unsigned int iterCount = 1;
    const unsigned int max_iter = dsolve::SOLVER_INIT_GRID_ITER;
    do {
#ifdef RK_SOLVER_OVERLAP_COMM_AND_COMP
        unzipVars_async(m_uiPrevVar, m_uiUnzipVar);
#else
        performGhostExchangeVars(m_uiPrevVar);
        unzipVars(m_uiPrevVar, m_uiUnzipVar);
#endif

        if (dsolve::SOLVER_ENABLE_BLOCK_ADAPTIVITY)
            isRefine = false;
        else {
            if (dsolve::SOLVER_REFINEMENT_MODE ==
                dsolve::RefinementMode::WAMR) {
                isRefine = m_uiMesh->isReMeshUnzip(
                    (const double **)m_uiUnzipVar, refineVarIds, refineNumVars,
                    waveletTolFunc, dsolve::SOLVER_DENDRO_AMR_FAC);
                // isRefine = dsolve::isReMeshWAMR(
                //     m_uiMesh, (const double **)m_uiUnzipVar, refineVarIds,
                //     refineNumVars, waveletTolFunc,
                //     dsolve::SOLVER_DENDRO_AMR_FAC);
            } else {
                std::cout << " Error : " << __func__
                          << " invalid refinement mode specified " << std::endl;
                MPI_Abort(m_uiComm, 0);
            }
        }

        if (isRefine) {
            ot::Mesh *newMesh = m_uiMesh->ReMesh(dsolve::SOLVER_DENDRO_GRAIN_SZ,
                                                 dsolve::SOLVER_LOAD_IMB_TOL,
                                                 dsolve::SOLVER_SPLIT_FIX);

            oldElements = m_uiMesh->getNumLocalMeshElements();
            newElements = newMesh->getNumLocalMeshElements();

            oldGridPoints = m_uiMesh->getNumLocalMeshNodes();
            newGridPoints = newMesh->getNumLocalMeshNodes();

            par::Mpi_Allreduce(&oldElements, &oldElements_g, 1, MPI_SUM,
                               m_uiMesh->getMPIGlobalCommunicator());
            par::Mpi_Allreduce(&newElements, &newElements_g, 1, MPI_SUM,
                               m_uiMesh->getMPIGlobalCommunicator());

            par::Mpi_Allreduce(&oldGridPoints, &oldGridPoints_g, 1, MPI_SUM,
                               m_uiMesh->getMPIGlobalCommunicator());
            par::Mpi_Allreduce(&newGridPoints, &newGridPoints_g, 1, MPI_SUM,
                               m_uiMesh->getMPIGlobalCommunicator());

            if (!(m_uiMesh->getMPIRankGlobal()))
                std::cout << "initial grid iteration : " << iterCount
                          << " old mesh (ele): " << oldElements_g
                          << " new mesh(ele): " << newElements_g << std::endl;
            if (!(m_uiMesh->getMPIRankGlobal()))
                std::cout << "initial grid iteration : " << iterCount
                          << " old mesh (zip nodes): " << oldGridPoints_g
                          << " new mesh(zip nodes): " << newGridPoints_g
                          << std::endl;

            // performs the inter-grid transfer
            intergridTransferVars(m_uiPrevVar, newMesh);

            for (unsigned int index = 0; index < dsolve::SOLVER_NUM_VARS;
                 index++) {
                delete[] m_uiVar[index];
                delete[] m_uiVarIm[index];
                delete[] m_uiUnzipVar[index];
                delete[] m_uiUnzipVarRHS[index];

                m_uiVar[index] = NULL;
                m_uiVarIm[index] = NULL;
                m_uiUnzipVar[index] = NULL;
                m_uiUnzipVarRHS[index] = NULL;

                m_uiVar[index] = newMesh->createVector<DendroScalar>();
                m_uiVarIm[index] = newMesh->createVector<DendroScalar>();
                m_uiUnzipVar[index] =
                    newMesh->createUnZippedVector<DendroScalar>();
                m_uiUnzipVarRHS[index] =
                    newMesh->createUnZippedVector<DendroScalar>();
            }

            for (unsigned int stage = 0; stage < m_uiNumRKStages; stage++)
                for (unsigned int index = 0; index < dsolve::SOLVER_NUM_VARS;
                     index++) {
                    delete[] m_uiStage[stage][index];
                    m_uiStage[stage][index] = NULL;
                    m_uiStage[stage][index] =
                        newMesh->createVector<DendroScalar>();
                }

            // deallocate constraint vars allocate them for the new mesh.
            for (unsigned int index = 0;
                 index < dsolve::SOLVER_CONSTRAINT_NUM_VARS; index++) {
                delete[] m_uiConstraintVars[index];
                delete[] m_uiUnzipConstraintVars[index];

                m_uiConstraintVars[index] =
                    newMesh->createVector<DendroScalar>();
                m_uiUnzipConstraintVars[index] =
                    newMesh->createUnZippedVector<DendroScalar>();
            }

            std::swap(newMesh, m_uiMesh);
            delete newMesh;

#ifdef RK_SOLVER_OVERLAP_COMM_AND_COMP
            // reallocates mpi resources for the the new mesh. (this will
            // deallocate the old resources)
            reallocateMPIResources();
#endif

            if (m_uiMesh->isActive()) {
                if (!(m_uiMesh->getMPIRank())) {
                    std::cout << "TRANSFER COMPLETED, VARIABLE SUMMARY FOR "
                              << m_uiMesh->getNumLocalMeshNodes()
                              << " NODES:" << std::endl;
                }

                for (const auto tmp_varID : dsolve::SOLVER_VAR_ITERABLE_LIST) {
                    DendroScalar l_min = vecMin(
                        m_uiPrevVar[tmp_varID] + m_uiMesh->getNodeLocalBegin(),
                        (m_uiMesh->getNumLocalMeshNodes()),
                        m_uiMesh->getMPICommunicator());
                    DendroScalar l_max = vecMax(
                        m_uiPrevVar[tmp_varID] + m_uiMesh->getNodeLocalBegin(),
                        (m_uiMesh->getNumLocalMeshNodes()),
                        m_uiMesh->getMPICommunicator());
                    if (!(m_uiMesh->getMPIRank())) {
                        std::cout << "    ||VAR::"
                                  << dsolve::SOLVER_VAR_NAMES[tmp_varID]
                                  << "|| (min, max) : (" << l_min << ", "
                                  << l_max << " ) " << std::endl;
                    }
                }
                // DendroScalar l_min = vecMin(m_uiPrevVar[dsolve::VAR::U_ALPHA]
                // +
                //                                 m_uiMesh->getNodeLocalBegin(),
                //                             (m_uiMesh->getNumLocalMeshNodes()),
                //                             m_uiMesh->getMPICommunicator());
                // DendroScalar l_max = vecMax(m_uiPrevVar[dsolve::VAR::U_ALPHA]
                // +
                //                                 m_uiMesh->getNodeLocalBegin(),
                //                             (m_uiMesh->getNumLocalMeshNodes()),
                //                             m_uiMesh->getMPICommunicator());
                // if (!(m_uiMesh->getMPIRank())) {
                //     std::cout << "transfer completed:    ||VAR::U_ALPHA|| "
                //                  "(min, max) : ("
                //               << l_min << ", " << l_max << " ) " <<
                //               std::endl;
                // }
            }

            iterCount += 1;
        }

    } while (isRefine &&
             (newElements_g != oldElements_g ||
              newGridPoints_g != oldGridPoints_g) &&
             (iterCount < max_iter));

    applyInitialConditions(m_uiPrevVar);

    // allocate the derivative variables
    dsolve::deallocate_deriv_workspace();
    dsolve::allocate_deriv_workspace(m_uiMesh, 1);

    unsigned int lmin, lmax;
    m_uiMesh->computeMinMaxLevel(lmin, lmax);
    dsolve::SOLVER_RK45_TIME_STEP_SIZE =
        dsolve::SOLVER_CFL_FACTOR *
        ((dsolve::SOLVER_COMPD_MAX[0] - dsolve::SOLVER_COMPD_MIN[0]) *
         ((1u << (m_uiMaxDepth - lmax)) / ((double)dsolve::SOLVER_ELE_ORDER)) /
         ((double)(1u << (m_uiMaxDepth))));
    m_uiT_h = dsolve::SOLVER_RK45_TIME_STEP_SIZE;
    if (!m_uiMesh->getMPIRankGlobal()) {
        std::cout << "================= Grid Info (After init grid "
                     "converge):==============================================="
                     "========"
                  << std::endl;
        std::cout << "lmin: " << lmin << " lmax:" << lmax << std::endl;
        std::cout << "dx: "
                  << ((dsolve::SOLVER_COMPD_MAX[0] -
                       dsolve::SOLVER_COMPD_MIN[0]) *
                      ((1u << (m_uiMaxDepth - lmax)) /
                       ((double)dsolve::SOLVER_ELE_ORDER)) /
                      ((double)(1u << (m_uiMaxDepth))))
                  << std::endl;
        std::cout << "dt: "
                  << dsolve::SOLVER_CFL_FACTOR *
                         ((dsolve::SOLVER_COMPD_MAX[0] -
                           dsolve::SOLVER_COMPD_MIN[0]) *
                          ((1u << (m_uiMaxDepth - lmax)) /
                           ((double)dsolve::SOLVER_ELE_ORDER)) /
                          ((double)(1u << (m_uiMaxDepth))))
                  << std::endl;
        std::cout << "========================================================="
                     "======================================================"
                  << std::endl;
    }
}

void RK_SOLVER::reallocateMPIResources() {
    for (unsigned int index = 0; index < dsolve::SOLVER_ASYNC_COMM_K; index++) {
        delete[] m_uiSendNodeBuf[index];
        delete[] m_uiRecvNodeBuf[index];

        delete[] m_uiSendReqs[index];
        delete[] m_uiRecvReqs[index];

        delete[] m_uiSendSts[index];
        delete[] m_uiRecvSts[index];
    }

    for (unsigned int index = 0; index < dsolve::SOLVER_ASYNC_COMM_K; index++) {
        m_uiSendNodeBuf[index] = NULL;
        m_uiRecvNodeBuf[index] = NULL;

        m_uiSendReqs[index] = NULL;
        m_uiRecvReqs[index] = NULL;
        m_uiSendSts[index] = NULL;
        m_uiRecvSts[index] = NULL;
    }

    if (m_uiMesh->isActive()) {
        // allocate mpi comm. reqs and status
        for (unsigned int index = 0; index < dsolve::SOLVER_ASYNC_COMM_K;
             index++) {
            if (m_uiMesh->getGhostExcgTotalSendNodeCount() != 0)
                m_uiSendNodeBuf[index] =
                    new DendroScalar[m_uiMesh
                                         ->getGhostExcgTotalSendNodeCount()];
            if (m_uiMesh->getGhostExcgTotalRecvNodeCount() != 0)
                m_uiRecvNodeBuf[index] =
                    new DendroScalar[m_uiMesh
                                         ->getGhostExcgTotalRecvNodeCount()];

            if (m_uiMesh->getSendProcListSize() != 0) {
                m_uiSendReqs[index] =
                    new MPI_Request[m_uiMesh->getSendProcListSize()];
                m_uiSendSts[index] =
                    new MPI_Status[m_uiMesh->getSendProcListSize()];
            }

            if (m_uiMesh->getRecvProcListSize() != 0) {
                m_uiRecvReqs[index] =
                    new MPI_Request[m_uiMesh->getRecvProcListSize()];
                m_uiRecvSts[index] =
                    new MPI_Status[m_uiMesh->getRecvProcListSize()];
            }
        }
    }
}

void RK_SOLVER::writeToVTU(DendroScalar **evolZipVarIn,
                           DendroScalar **constrZipVarIn,
                           unsigned int numEvolVars, unsigned int numConstVars,
                           const unsigned int *evolVarIndices,
                           const unsigned int *constVarIndices, bool zslice) {
    dsolve::timer::t_ioVtu.start();

    std::vector<std::string> pDataNames;
    double *pData[(numConstVars + numEvolVars)];
    // ADDED TO CALCULATE THE ANAYLTIC DIFFERENCE WITH THE CFD'S -AJC
    double **diff = new double *[dsolve::SOLVER_NUM_VARS];
    for (unsigned int v = 0; v < dsolve::SOLVER_NUM_VARS; v++)
        diff[v] = m_uiMesh->createVector<double>(0.0);

    // initialize diff begin.
    unsigned int nodeLookUp_CG;
    unsigned int nodeLookUp_DG;
    double x, y, z, len;
    const ot::TreeNode *pNodes = &(*(m_uiMesh->getAllElements().begin()));
    unsigned int ownerID, ii_x, jj_y, kk_z;
    unsigned int eleOrder = m_uiMesh->getElementOrder();
    const unsigned int *e2n_cg = &(*(m_uiMesh->getE2NMapping().begin()));
    const unsigned int *e2n_dg = &(*(m_uiMesh->getE2NMapping_DG().begin()));
    const unsigned int nPe = m_uiMesh->getNumNodesPerElement();
    const unsigned int nodeLocalBegin = m_uiMesh->getNodeLocalBegin();
    const unsigned int nodeLocalEnd = m_uiMesh->getNodeLocalEnd();

    double var[dsolve::SOLVER_NUM_VARS];
    for (unsigned int elem = m_uiMesh->getElementLocalBegin();
         elem < m_uiMesh->getElementLocalEnd(); elem++) {
        for (unsigned int k = 0; k < (eleOrder + 1); k++)
            for (unsigned int j = 0; j < (eleOrder + 1); j++)
                for (unsigned int i = 0; i < (eleOrder + 1); i++) {
                    nodeLookUp_CG = e2n_cg[elem * nPe +
                                           k * (eleOrder + 1) * (eleOrder + 1) +
                                           j * (eleOrder + 1) + i];
                    if (nodeLookUp_CG >= nodeLocalBegin &&
                        nodeLookUp_CG < nodeLocalEnd) {
                        nodeLookUp_DG =
                            e2n_dg[elem * nPe +
                                   k * (eleOrder + 1) * (eleOrder + 1) +
                                   j * (eleOrder + 1) + i];
                        m_uiMesh->dg2eijk(nodeLookUp_DG, ownerID, ii_x, jj_y,
                                          kk_z);
                        len = (double)(1u << (m_uiMaxDepth -
                                              pNodes[ownerID].getLevel()));
                        x = pNodes[ownerID].getX() + ii_x * (len / (eleOrder));
                        y = pNodes[ownerID].getY() + jj_y * (len / (eleOrder));
                        z = pNodes[ownerID].getZ() + kk_z * (len / (eleOrder));

                        dsolve::analyticalSolEM4((double)x, (double)y,
                                                 (double)z, m_uiCurrentTime,
                                                 var);
                        for (unsigned int v = 0; v < dsolve::SOLVER_NUM_VARS;
                             v++)
                            diff[v][nodeLookUp_CG] =
                                evolZipVarIn[v][nodeLookUp_CG] - var[v];
                    }
                }
    }

    // initialize diff end

    for (unsigned int i = 0; i < numEvolVars; i++) {
        pDataNames.push_back(
            std::string(dsolve::SOLVER_VAR_NAMES[evolVarIndices[i]]));
        pData[i] = evolZipVarIn[evolVarIndices[i]];
    }

    for (unsigned int i = 0; i < numConstVars; i++) {
        pDataNames.push_back(std::string(
            dsolve::SOLVER_VAR_CONSTRAINT_NAMES[constVarIndices[i]]));
        pData[numEvolVars + i] = constrZipVarIn[constVarIndices[i]];
    }

    std::vector<char *> pDataNames_char;
    pDataNames_char.reserve(pDataNames.size());

    for (unsigned int i = 0; i < pDataNames.size(); i++)
        pDataNames_char.push_back(const_cast<char *>(pDataNames[i].c_str()));

    const char *fDataNames[] = {"Time", "Cycle"};
    const double fData[] = {m_uiCurrentTime, (double)m_uiCurrentStep};

    char fPrefix[256];
    sprintf(fPrefix, "%s_%d", dsolve::SOLVER_VTU_FILE_PREFIX.c_str(),
            m_uiCurrentStep);

    if (zslice) {
        unsigned int s_val[3] = {1u << (m_uiMaxDepth - 1),
                                 1u << (m_uiMaxDepth - 1),
                                 1u << (m_uiMaxDepth - 1)};
        unsigned int s_norm[3] = {0, 0, 1};
        io::vtk::mesh2vtu_slice(m_uiMesh, s_val, s_norm, fPrefix, 2, fDataNames,
                                fData, (numEvolVars + numConstVars),
                                (const char **)&pDataNames_char[0],
                                (const double **)pData);
    } else {
        io::vtk::mesh2vtuFine(m_uiMesh, fPrefix, 2, fDataNames, fData,
                              (numEvolVars + numConstVars),
                              (const char **)&pDataNames_char[0],
                              (const double **)pData);
    }

    dsolve::timer::t_ioVtu.stop();
}

void RK_SOLVER::writeEvolutionAndRHStoVTU(double **evolZipVarIn,
                                          double **evolRHSZipVarIn,
                                          unsigned int numEvolVars,
                                          const unsigned int *evolVarIndices,
                                          const unsigned int stage) {
    if (!m_uiMesh->isActive()) return;

    dsolve::timer::t_ioVtu.start();

    std::vector<std::string> pDataNames;
    double *pData[(numEvolVars * 2)];

    // gather the evolution variables we want to output
    for (unsigned int i = 0; i < numEvolVars; i++) {
        pDataNames.push_back(
            std::string(dsolve::SOLVER_VAR_NAMES[evolVarIndices[i]]));
        pData[i] = evolZipVarIn[evolVarIndices[i]];
    }

    // gather the evolution RHS variables we want to output
    for (unsigned int i = 0; i < numEvolVars; i++) {
        pDataNames.push_back(
            std::string(dsolve::SOLVER_VAR_NAMES[evolVarIndices[i]]) + "_RHS");
        pData[i + numEvolVars] = evolRHSZipVarIn[evolVarIndices[i]];
    }

    std::vector<char *> pDataNames_char;
    pDataNames_char.reserve(pDataNames.size());

    for (unsigned int i = 0; i < pDataNames.size(); i++)
        pDataNames_char.push_back(const_cast<char *>(pDataNames[i].c_str()));

    const char *fDataNames[] = {"Time", "Cycle"};
    const double fData[] = {m_uiCurrentTime, (double)m_uiCurrentStep};

    char fPrefix[256];
    sprintf(fPrefix, "%s_evoRHS_%d_s%d", dsolve::SOLVER_VTU_FILE_PREFIX.c_str(),
            stage, m_uiCurrentStep);

    io::vtk::mesh2vtuFine(m_uiMesh, fPrefix, 2, fDataNames, fData,
                          (numEvolVars * 2), (const char **)&pDataNames_char[0],
                          (const double **)pData);

    dsolve::timer::t_ioVtu.stop();
}

void RK_SOLVER::performGhostExchangeVars(DendroScalar **zipIn) {
    dsolve::timer::t_ghostEx_sync.start();

    for (unsigned int v = 0; v < dsolve::SOLVER_NUM_VARS; v++)
        m_uiMesh->performGhostExchange(zipIn[v]);

    dsolve::timer::t_ghostEx_sync.stop();
}

void RK_SOLVER::intergridTransferVars(DendroScalar **&zipIn,
                                      const ot::Mesh *pnewMesh) {
    dsolve::timer::t_gridTransfer.start();

    for (unsigned int v = 0; v < dsolve::SOLVER_NUM_VARS; v++)
        m_uiMesh->interGridTransfer(zipIn[v], pnewMesh);

    dsolve::timer::t_gridTransfer.stop();
}

void RK_SOLVER::unzipVars(DendroScalar **zipIn, DendroScalar **uzipOut) {
    dsolve::timer::t_unzip_sync.start();

    for (unsigned int index = 0; index < dsolve::SOLVER_NUM_VARS; index++)
        m_uiMesh->unzip(zipIn[index], uzipOut[index]);

    dsolve::timer::t_unzip_sync.stop();
}

void RK_SOLVER::unzipVars_async(DendroScalar **zipIn, DendroScalar **uzipOut) {
    dsolve::timer::t_unzip_async.start();

    for (unsigned int var = 0; var < dsolve::SOLVER_NUM_VARS;
         var += dsolve::SOLVER_ASYNC_COMM_K) {
        for (unsigned int i = 0; (i < dsolve::SOLVER_ASYNC_COMM_K); i++)
            m_uiMesh->ghostExchangeStart(zipIn[var + i], m_uiSendNodeBuf[i],
                                         m_uiRecvNodeBuf[i], m_uiSendReqs[i],
                                         m_uiRecvReqs[i]);

        for (unsigned int i = 0; (i < dsolve::SOLVER_ASYNC_COMM_K); i++) {
            m_uiMesh->ghostExchangeRecvSync(zipIn[var + i], m_uiRecvNodeBuf[i],
                                            m_uiRecvReqs[i], m_uiRecvSts[i]);
            m_uiMesh->unzip(zipIn[var + i], uzipOut[var + i]);
        }

        for (unsigned int i = 0; (i < dsolve::SOLVER_ASYNC_COMM_K); i++)
            m_uiMesh->ghostExchangeSendSync(m_uiSendReqs[i], m_uiSendSts[i]);
    }

    dsolve::timer::t_unzip_async.stop();
}

void RK_SOLVER::zipVars(DendroScalar **uzipIn, DendroScalar **zipOut) {
    dsolve::timer::t_zip.start();

    for (unsigned int index = 0; index < dsolve::SOLVER_NUM_VARS; index++)
        m_uiMesh->zip(uzipIn[index], zipOut[index]);

    dsolve::timer::t_zip.stop();
}

void RK_SOLVER::applyBoundaryConditions() {}

void RK_SOLVER::performSingleIterationRK3() {
    // BEGIN COMMON DEFINITIONS AND OPERATIONS NECESSARY FOR EACH RK TYPE
    char frawName[256];

    double current_t = m_uiCurrentTime;
    double current_t_adv = current_t;

#ifdef RK_SOLVER_OVERLAP_COMM_AND_COMP
    // async unzip to assign the "prevVar" values to the unzip variable
    unzipVars_async(m_uiPrevVar, m_uiUnzipVar);
#else
    // ghost exchange and then unzip if there is no overlap enabled
    performGhostExchangeVars(m_uiPrevVar);
    unzipVars(m_uiPrevVar, m_uiUnzipVar);
#endif

    int rank = m_uiMesh->getMPIRank();

    const unsigned int nodeLocalBegin = m_uiMesh->getNodeLocalBegin();
    const unsigned int nodeLocalEnd = m_uiMesh->getNodeLocalEnd();

    const std::vector<ot::Block> &blkList = m_uiMesh->getLocalBlockList();
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

    // END COMMON DEFINITIONS AND OPERATIONS

    // initial unzip and ghost exchange happens at the rkSolve class.
    solverRHS(m_uiUnzipVarRHS, (const DendroScalar **)m_uiUnzipVar,
              &(*(blkList.begin())), blkList.size());
    zipVars(m_uiUnzipVarRHS, m_uiStage[0]);

    for (unsigned int node = nodeLocalBegin; node < nodeLocalEnd; node++) {
        // if ( node == 101 ) {
        // std::cout << "yo 2a node=" << node << std::endl ;
        // std::cout << "At_xx = " << m_uiStage[0][9][node] << std::endl ;
        // std::cout << "At_xy = " << m_uiStage[0][10][node] << std::endl ;
        // std::cout << "At_xz = " << m_uiStage[0][11][node] << std::endl ;
        // std::cout << "At_yy = " << m_uiStage[0][12][node] << std::endl ;
        // std::cout << "At_yz = " << m_uiStage[0][13][node] << std::endl ;
        // std::cout << "At_zz = " << m_uiStage[0][14][node] << std::endl ;
        //}

        for (unsigned int index = 0; index < dsolve::SOLVER_NUM_VARS; index++) {
            m_uiStage[0][index][node] =
                m_uiPrevVar[index][node] + m_uiT_h * m_uiStage[0][index][node];
        }

        // if ( node == 101 ) {
        // std::cout << "yo 2b node=" << node << std::endl ;
        // std::cout << "At_xx = " << m_uiPrevVar[9][node] << std::endl ;
        // std::cout << "At_xy = " << m_uiPrevVar[10][node] << std::endl ;
        // std::cout << "At_xz = " << m_uiPrevVar[11][node] << std::endl ;
        // std::cout << "At_yy = " << m_uiPrevVar[12][node] << std::endl ;
        // std::cout << "At_yz = " << m_uiPrevVar[13][node] << std::endl ;
        // std::cout << "At_zz = " << m_uiPrevVar[14][node] << std::endl ;
        //
        // std::cout << "At_xx = " << m_uiStage[0][9][node] << std::endl ;
        // std::cout << "At_xy = " << m_uiStage[0][10][node] << std::endl ;
        // std::cout << "At_xz = " << m_uiStage[0][11][node] << std::endl ;
        // std::cout << "At_yy = " << m_uiStage[0][12][node] << std::endl ;
        // std::cout << "At_yz = " << m_uiStage[0][13][node] << std::endl ;
        // std::cout << "At_zz = " << m_uiStage[0][14][node] << std::endl ;
        //}

        enforce_system_constraints(m_uiStage[0], node);
    }

#if 0            
            sprintf(frawName,"rkU_%d",3*m_uiCurrentStep + 1);
            io::varToRawData((const ot::Mesh*)m_uiMesh,(const double **)m_uiStage[0],dsolve::SOLVER_NUM_VARS,NULL,frawName);
#endif

#ifdef RK_SOLVER_OVERLAP_COMM_AND_COMP
    unzipVars_async(m_uiStage[0], m_uiUnzipVar);
#else
    performGhostExchangeVars(m_uiStage[0]);
    unzipVars(m_uiStage[0], m_uiUnzipVar);
#endif

    solverRHS(m_uiUnzipVarRHS, (const DendroScalar **)m_uiUnzipVar,
              &(*(blkList.begin())), blkList.size());
    zipVars(m_uiUnzipVarRHS, m_uiStage[1]);

    for (unsigned int node = nodeLocalBegin; node < nodeLocalEnd; node++) {
        // if ( node == 101 ) {
        // std::cout << "yo 3a node=" << node << std::endl ;
        // std::cout << "At_xx = " << m_uiStage[1][9][node] << std::endl ;
        // std::cout << "At_xy = " << m_uiStage[1][10][node] << std::endl ;
        // std::cout << "At_xz = " << m_uiStage[1][11][node] << std::endl ;
        // std::cout << "At_yy = " << m_uiStage[1][12][node] << std::endl ;
        // std::cout << "At_yz = " << m_uiStage[1][13][node] << std::endl ;
        // std::cout << "At_zz = " << m_uiStage[1][14][node] << std::endl ;
        //}

        for (unsigned int index = 0; index < dsolve::SOLVER_NUM_VARS; index++) {
            m_uiStage[1][index][node] =
                (0.75) * m_uiPrevVar[index][node] +
                0.25 * m_uiStage[0][index][node] +
                m_uiT_h * 0.25 * m_uiStage[1][index][node];
        }

        // if ( node == 101 ) {
        // std::cout << "yo 3b node=" << node << std::endl ;
        // std::cout << "At_xx = " << m_uiPrevVar[9][node] << std::endl ;
        // std::cout << "At_xy = " << m_uiPrevVar[10][node] << std::endl ;
        // std::cout << "At_xz = " << m_uiPrevVar[11][node] << std::endl ;
        // std::cout << "At_yy = " << m_uiPrevVar[12][node] << std::endl ;
        // std::cout << "At_yz = " << m_uiPrevVar[13][node] << std::endl ;
        // std::cout << "At_zz = " << m_uiPrevVar[14][node] << std::endl ;
        //
        // std::cout << "At_xx = " << m_uiStage[0][9][node] << std::endl ;
        // std::cout << "At_xy = " << m_uiStage[0][10][node] << std::endl ;
        // std::cout << "At_xz = " << m_uiStage[0][11][node] << std::endl ;
        // std::cout << "At_yy = " << m_uiStage[0][12][node] << std::endl ;
        // std::cout << "At_yz = " << m_uiStage[0][13][node] << std::endl ;
        // std::cout << "At_zz = " << m_uiStage[0][14][node] << std::endl ;
        //
        // std::cout << "At_xx = " << m_uiStage[1][9][node] << std::endl ;
        // std::cout << "At_xy = " << m_uiStage[1][10][node] << std::endl ;
        // std::cout << "At_xz = " << m_uiStage[1][11][node] << std::endl ;
        // std::cout << "At_yy = " << m_uiStage[1][12][node] << std::endl ;
        // std::cout << "At_yz = " << m_uiStage[1][13][node] << std::endl ;
        // std::cout << "At_zz = " << m_uiStage[1][14][node] << std::endl ;
        //}

        enforce_system_constraints(m_uiStage[1], node);
    }

#if 0
            sprintf(frawName,"rkU_%d",3*m_uiCurrentStep + 2);
            io::varToRawData((const ot::Mesh*)m_uiMesh,(const double **)m_uiStage[1],dsolve::SOLVER_NUM_VARS,NULL,frawName);
#endif

#ifdef RK_SOLVER_OVERLAP_COMM_AND_COMP
    unzipVars_async(m_uiStage[1], m_uiUnzipVar);
#else
    performGhostExchangeVars(m_uiStage[1]);
    unzipVars(m_uiStage[1], m_uiUnzipVar);
#endif

    solverRHS(m_uiUnzipVarRHS, (const DendroScalar **)m_uiUnzipVar,
              &(*(blkList.begin())), blkList.size());

    for (unsigned int node = nodeLocalBegin; node < nodeLocalEnd; node++) {
        // if ( node == 101 ) {
        //
        // std::cout << "yo 4aa node=" << node << std::endl ;
        // std::cout << "rhsAt_xx = " << m_uiUnzipVarRHS[9][node] << std::endl;
        // std::cout << "rhsAt_xy = " << m_uiUnzipVarRHS[10][node] << std::endl;
        // std::cout << "rhsAt_xz = " << m_uiUnzipVarRHS[11][node] << std::endl;
        // std::cout << "rhsAt_yy = " << m_uiUnzipVarRHS[12][node] << std::endl;
        // std::cout << "rhsAt_yz = " << m_uiUnzipVarRHS[13][node] << std::endl;
        // std::cout << "rhsAt_zz = " << m_uiUnzipVarRHS[14][node] << std::endl;
        //
        //}
    }

    zipVars(m_uiUnzipVarRHS, m_uiVar);

    for (unsigned int node = nodeLocalBegin; node < nodeLocalEnd; node++) {
        // if ( node == 101 ) {
        // std::cout << "yo 4a node=" << node << std::endl ;
        // std::cout << "rhsAt_xx = " << m_uiUnzipVarRHS[9][node] << std::endl;
        // std::cout << "rhsAt_xy = " << m_uiUnzipVarRHS[10][node] << std::endl;
        // std::cout << "rhsAt_xz = " << m_uiUnzipVarRHS[11][node] << std::endl;
        // std::cout << "rhsAt_yy = " << m_uiUnzipVarRHS[12][node] << std::endl;
        // std::cout << "rhsAt_yz = " << m_uiUnzipVarRHS[13][node] << std::endl;
        // std::cout << "rhsAt_zz = " << m_uiUnzipVarRHS[14][node] << std::endl;
        //
        // std::cout << "At_xx = " << m_uiVar[9][node] << std::endl ;
        // std::cout << "At_xy = " << m_uiVar[10][node] << std::endl ;
        // std::cout << "At_xz = " << m_uiVar[11][node] << std::endl ;
        // std::cout << "At_yy = " << m_uiVar[12][node] << std::endl ;
        // std::cout << "At_yz = " << m_uiVar[13][node] << std::endl ;
        // std::cout << "At_zz = " << m_uiVar[14][node] << std::endl ;
        //}

        for (unsigned int index = 0; index < dsolve::SOLVER_NUM_VARS; index++) {
            m_uiVar[index][node] = (1.0 / 3.0) * m_uiPrevVar[index][node] +
                                   (2.0 / 3.0) * m_uiStage[1][index][node] +
                                   m_uiT_h * (2.0 / 3.0) * m_uiVar[index][node];
        }

        // if ( node == 101 ) {
        // std::cout << "yo 4b node=" << node << std::endl ;
        ////std::cout << "m_uiT_h = " << m_uiT_h << std::endl ;
        // std::cout << "prev_At_xx = " << m_uiPrevVar[ 9][node] << std::endl ;
        // std::cout << "prev_At_xy = " << m_uiPrevVar[10][node] << std::endl ;
        // std::cout << "prev_At_xz = " << m_uiPrevVar[11][node] << std::endl ;
        // std::cout << "prev_At_yy = " << m_uiPrevVar[12][node] << std::endl ;
        // std::cout << "prev_At_yz = " << m_uiPrevVar[13][node] << std::endl ;
        // std::cout << "prev_At_zz = " << m_uiPrevVar[14][node] << std::endl ;
        //
        // std::cout << "stgd_At_xx = " << m_uiStage[1][9][node] << std::endl ;
        // std::cout << "stgd_At_xy = " << m_uiStage[1][10][node] << std::endl ;
        // std::cout << "stgd_At_xz = " << m_uiStage[1][11][node] << std::endl ;
        // std::cout << "stgd_At_yy = " << m_uiStage[1][12][node] << std::endl ;
        // std::cout << "stgd_At_yz = " << m_uiStage[1][13][node] << std::endl ;
        // std::cout << "stgd_At_zz = " << m_uiStage[1][14][node] << std::endl ;
        //
        // std::cout << "At_xx = " << m_uiVar[9][node] << std::endl ;
        // std::cout << "At_xy = " << m_uiVar[10][node] << std::endl ;
        // std::cout << "At_xz = " << m_uiVar[11][node] << std::endl ;
        // std::cout << "At_yy = " << m_uiVar[12][node] << std::endl ;
        // std::cout << "At_yz = " << m_uiVar[13][node] << std::endl ;
        // std::cout << "At_zz = " << m_uiVar[14][node] << std::endl ;
        //
        // std::cout << "gt_xx = " << m_uiVar[15][node] << std::endl ;
        // std::cout << "gt_xy = " << m_uiVar[16][node] << std::endl ;
        // std::cout << "gt_xz = " << m_uiVar[17][node] << std::endl ;
        // std::cout << "gt_yy = " << m_uiVar[18][node] << std::endl ;
        // std::cout << "gt_yz = " << m_uiVar[19][node] << std::endl ;
        // std::cout << "gt_zz = " << m_uiVar[20][node] << std::endl ;
        //}

        enforce_system_constraints(m_uiVar, node);
    }

    /*
*
*  !!!! this does not work for some reason .
// stage 0   f(u_k)
solverRHS(m_uiUnzipVarRHS,(const DendroScalar
**)m_uiUnzipVar,&(*(blkList.begin())),blkList.size());
zipVars(m_uiUnzipVarRHS,m_uiStage[0]);
#if 0
sprintf(frawName,"rk3_step_%d_stage_%d",m_uiCurrentStep,0);
io::varToRawData((const ot::Mesh*)m_uiMesh,(const double
**)m_uiStage[0],dsolve::SOLVER_NUM_VARS,NULL,frawName); #endif
// u1
for(unsigned int node=nodeLocalBegin; node<nodeLocalEnd; node++)
{

for(unsigned int index=0; index<dsolve::SOLVER_NUM_VARS; index++)
{
   m_uiVarIm[index][node]=m_uiPrevVar[index][node] + m_uiT_h *
m_uiStage[0][index][node];
}
enforce_system_constraints(m_uiVarIm, node);
}

#if 0
sprintf(frawName,"rkU_%d",3*m_uiCurrentStep+(1));
io::varToRawData((const ot::Mesh*)m_uiMesh,(const double
**)m_uiVarIm,dsolve::SOLVER_NUM_VARS,NULL,frawName); #endif
current_t_adv=current_t+m_uiT_h;

#ifdef RK_SOLVER_OVERLAP_COMM_AND_COMP
unzipVars_async(m_uiVarIm,m_uiUnzipVar);
#else
performGhostExchangeVars(m_uiVarIm);
unzipVars(m_uiVarIm,m_uiUnzipVar);
#endif


// stage 1
solverRHS(m_uiUnzipVarRHS,(const DendroScalar
**)m_uiUnzipVar,&(*(blkList.begin())),blkList.size());
zipVars(m_uiUnzipVarRHS,m_uiStage[1]);
#if 0
sprintf(frawName,"rk3_step_%d_stage_%d",m_uiCurrentStep,1);
io::varToRawData((const ot::Mesh*)m_uiMesh,(const double
**)m_uiStage[1],dsolve::SOLVER_NUM_VARS,NULL,frawName); #endif

// u2
for(unsigned int node=nodeLocalBegin; node<nodeLocalEnd; node++)
{

for(unsigned int index=0; index<dsolve::SOLVER_NUM_VARS; index++)
{
   m_uiVarIm[index][node]=m_uiPrevVar[index][node] + m_uiT_h * 0.25 *
(m_uiStage[0][index][node] + m_uiStage[1][index][node]);
}
enforce_system_constraints(m_uiVarIm, node);
}

#if 0
sprintf(frawName,"rkU_%d",3*m_uiCurrentStep+(2));
io::varToRawData((const ot::Mesh*)m_uiMesh,(const double
**)m_uiVarIm,dsolve::SOLVER_NUM_VARS,NULL,frawName); #endif
current_t_adv=current_t+m_uiT_h;

#ifdef RK_SOLVER_OVERLAP_COMM_AND_COMP
unzipVars_async(m_uiVarIm,m_uiUnzipVar);
#else
performGhostExchangeVars(m_uiVarIm);
unzipVars(m_uiVarIm,m_uiUnzipVar);
#endif

// stage 2.
solverRHS(m_uiUnzipVarRHS,(const DendroScalar
**)m_uiUnzipVar,&(*(blkList.begin())),blkList.size());
zipVars(m_uiUnzipVarRHS,m_uiStage[2]);

#if 0
sprintf(frawName,"rk3_step_%d_stage_%d",m_uiCurrentStep,2);
io::varToRawData((const ot::Mesh*)m_uiMesh,(const double
**)m_uiStage[1],dsolve::SOLVER_NUM_VARS,NULL,frawName); #endif

// u_(k+1)
for(unsigned int node=nodeLocalBegin; node<nodeLocalEnd; node++)
{

for(unsigned int index=0; index<dsolve::SOLVER_NUM_VARS; index++)
{
   m_uiVar[index][node]=m_uiPrevVar[index][node] + m_uiT_h *(
(1.0/6.0)*m_uiStage[0][index][node] + (1.0/6.0)*m_uiStage[1][index][node] +
(2.0/3.0)*m_uiStage[2][index][node]);
}
enforce_system_constraints(m_uiVar, node);
}
*/
}

void RK_SOLVER::performSingleIterationRK4() {
    // BEGIN COMMON DEFINITIONS AND OPERATIONS NECESSARY FOR EACH RK TYPE
    char frawName[256];

    double current_t = m_uiCurrentTime;
    double current_t_adv = current_t;

#ifdef RK_SOLVER_OVERLAP_COMM_AND_COMP
    // async unzip to assign the "prevVar" values to the unzip variable
    unzipVars_async(m_uiPrevVar, m_uiUnzipVar);
#else
    // ghost exchange and then unzip if there is no overlap enabled
    performGhostExchangeVars(m_uiPrevVar);
    unzipVars(m_uiPrevVar, m_uiUnzipVar);
#endif

    int rank = m_uiMesh->getMPIRank();

    const unsigned int nodeLocalBegin = m_uiMesh->getNodeLocalBegin();
    const unsigned int nodeLocalEnd = m_uiMesh->getNodeLocalEnd();

    const std::vector<ot::Block> &blkList = m_uiMesh->getLocalBlockList();
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

    // END COMMON DEFINITIONS AND OPERATIONS

    // BEGIN RK4 STEPPING

    // iterate through the variable RK4 stages to compute the next steps
    for (unsigned int stage = 0; stage < (dsolve::SOLVER_RK4_STAGES - 1);
         stage++) {
#ifdef DEBUG_RK_SOLVER
        if (!rank) std::cout << " stage: " << stage << " begin: " << std::endl;
        for (unsigned int index = 0; index < dsolve::SOLVER_NUM_VARS; index++)
            ot::test::isUnzipNaN(m_uiMesh, m_uiUnzipVar[index]);
#endif

        solverRHS(m_uiUnzipVarRHS, (const DendroScalar **)m_uiUnzipVar,
                  &(*(blkList.begin())), blkList.size());

#ifdef DEBUG_RK_SOLVER
        if (!rank)
            std::cout << " stage: " << stage
                      << " af rhs UNZIP RHS TEST:" << std::endl;
        for (unsigned int index = 0; index < dsolve::SOLVER_NUM_VARS; index++)
            ot::test::isUnzipInternalNaN(m_uiMesh, m_uiUnzipVarRHS[index]);
#endif

        // zip the calculated RHS variables into the "stage" variable
        zipVars(m_uiUnzipVarRHS, m_uiStage[stage]);

#ifdef DEBUG_RK_SOLVER
        for (unsigned int index = 0; index < dsolve::SOLVER_NUM_VARS; index++)
            if (seq::test::isNAN(
                    m_uiStage[stage][index] + m_uiMesh->getNodeLocalBegin(),
                    m_uiMesh->getNumLocalMeshNodes()))
                std::cout << " var: " << index
                          << " contains nan af zip  stage: " << stage
                          << std::endl;
#endif

        /*for(unsigned int index=0;index<dsolve::SOLVER_NUM_VARS;index++)
            for(unsigned int node=nodeLocalBegin;node<nodeLocalEnd;node++)
                m_uiStage[stage][index][node]*=m_uiT_h;*/

        // throughout the nodes we need to assign the "previous" variable
        // values to the intermediate step and enforce constraints
        for (unsigned int node = nodeLocalBegin; node < nodeLocalEnd; node++) {
            for (unsigned int index = 0; index < dsolve::SOLVER_NUM_VARS;
                 index++) {
                m_uiVarIm[index][node] = m_uiPrevVar[index][node];

                m_uiVarIm[index][node] += (RK4_U[stage + 1] * m_uiT_h *
                                           m_uiStage[stage][index][node]);
            }

            // if ( node == 101 ) {
            // std::cout << "yo 5 node=" << node << std::endl ;
            //}

            enforce_system_constraints(m_uiVarIm, node);
        }

#ifdef SOLVER_SAVE_RHS_EVERY_SINGLE_STEP
        // TEMP: save the output of the current variables!!!
        std::cout << "Now saving RHS portion" << std::endl;
        writeEvolutionAndRHStoVTU(m_uiPrevVar, m_uiStage[stage],
                                  dsolve::SOLVER_NUM_EVOL_VARS_VTU_OUTPUT,
                                  dsolve::SOLVER_VTU_OUTPUT_EVOL_INDICES,
                                  stage);
#endif

        // then we update the time
        current_t_adv = current_t + RK4_T[stage + 1] * m_uiT_h;
#ifdef RK_SOLVER_OVERLAP_COMM_AND_COMP
        unzipVars_async(m_uiVarIm, m_uiUnzipVar);
#else
        performGhostExchangeVars(m_uiVarIm);
        unzipVars(m_uiVarIm, m_uiUnzipVar);
#endif
    }  // END RK4 STAGE BLOCKS

    // update the time again
    current_t_adv =
        current_t + RK4_T[(dsolve::SOLVER_RK4_STAGES - 1)] * m_uiT_h;

#ifdef DEBUG_RK_SOLVER
    if (!rank)
        std::cout << " stage: " << (dsolve::SOLVER_RK4_STAGES - 1)
                  << " begin: " << std::endl;

    for (unsigned int index = 0; index < dsolve::SOLVER_NUM_VARS; index++)
        ot::test::isUnzipNaN(m_uiMesh, m_uiUnzipVar[index]);
#endif

    solverRHS(m_uiUnzipVarRHS, (const DendroScalar **)m_uiUnzipVar,
              &(*(blkList.begin())), blkList.size());

#ifdef DEBUG_RK_SOLVER
    if (!rank)
        std::cout << " stage: " << (dsolve::SOLVER_RK4_STAGES - 1)
                  << " af rhs UNZIP RHS TEST:" << std::endl;
    for (unsigned int index = 0; index < dsolve::SOLVER_NUM_VARS; index++)
        ot::test::isUnzipInternalNaN(m_uiMesh, m_uiUnzipVarRHS[index]);
#endif
    // put the unziped RHS values into the "stage" storage for the final RK4
    // stage
    zipVars(m_uiUnzipVarRHS, m_uiStage[(dsolve::SOLVER_RK4_STAGES - 1)]);

    // std::cout << "Running final step for RK4 solver with " << nodeLocalEnd <<
    // " nodes:" << std::endl;

    // update the variables with the calculated update based on RHS
    for (unsigned int node = nodeLocalBegin; node < nodeLocalEnd; node++) {
        for (unsigned int index = 0; index < dsolve::SOLVER_NUM_VARS; index++) {
            m_uiVar[index][node] = m_uiPrevVar[index][node];
            for (unsigned int s = 0; s < (dsolve::SOLVER_RK4_STAGES); s++) {
                m_uiVar[index][node] +=
                    (RK4_C[s] * m_uiT_h * m_uiStage[s][index][node]);
            }
        }

        // if ( node == 101 ) {
        // std::cout << "yo 6 node=" << node << std::endl ;
        //}

        // enforce constraints again just to make sure we didn't mess up
        enforce_system_constraints(m_uiVar, node);
    }

    // std::cout << "Finished RK4 Step!" << std::endl;
}

void RK_SOLVER::performSingleIterationRK45() {
    // BEGIN COMMON DEFINITIONS AND OPERATIONS NECESSARY FOR EACH RK TYPE
    char frawName[256];

    double current_t = m_uiCurrentTime;
    double current_t_adv = current_t;

#ifdef RK_SOLVER_OVERLAP_COMM_AND_COMP
    // async unzip to assign the "prevVar" values to the unzip variable
    unzipVars_async(m_uiPrevVar, m_uiUnzipVar);
#else
    // ghost exchange and then unzip if there is no overlap enabled
    performGhostExchangeVars(m_uiPrevVar);
    unzipVars(m_uiPrevVar, m_uiUnzipVar);
#endif

    int rank = m_uiMesh->getMPIRank();

    const unsigned int nodeLocalBegin = m_uiMesh->getNodeLocalBegin();
    const unsigned int nodeLocalEnd = m_uiMesh->getNodeLocalEnd();

    const std::vector<ot::Block> &blkList = m_uiMesh->getLocalBlockList();
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

    // END COMMON DEFINITIONS AND OPERATIONS

    // BEGIN RK45 STEPPING
    // std::cout<<"rk45"<<std::endl;

    bool repeatStep;
    double n_inf_max = 0.0;
    double n_inf_max_g = 0;

    do {
        repeatStep = false;
        n_inf_max = 0;
        n_inf_max_g = 0;

        if (m_uiMesh->isActive()) {
            double current_t = m_uiCurrentTime;
            double current_t_adv = current_t;
#ifdef RK_SOLVER_OVERLAP_COMM_AND_COMP
            unzipVars_async(m_uiPrevVar, m_uiUnzipVar);
#else
            // 1. perform ghost exchange.
            performGhostExchangeVars(m_uiPrevVar);

            // 2. unzip all the variables.
            unzipVars(m_uiPrevVar, m_uiUnzipVar);
#endif

            int rank = m_uiMesh->getMPIRank();

            const unsigned int nodeLocalBegin = m_uiMesh->getNodeLocalBegin();
            const unsigned int nodeLocalEnd = m_uiMesh->getNodeLocalEnd();

            const std::vector<ot::Block> &blkList =
                m_uiMesh->getLocalBlockList();

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

            for (unsigned int stage = 0;
                 stage < (dsolve::SOLVER_RK45_STAGES - 1); stage++) {
#ifdef DEBUG_RK_SOLVER
                if (!rank)
                    std::cout << " stage: " << stage << " begin: " << std::endl;
                for (unsigned int index = 0; index < dsolve::SOLVER_NUM_VARS;
                     index++)
                    ot::test::isUnzipNaN(m_uiMesh, m_uiUnzipVar[index]);
#endif

#ifdef SOLVER_ENABLE_CUDA
                cuda::SOLVERComputeParams solverParams;
                solverParams.SOLVER_LAMBDA[0] = dsolve::SOLVER_LAMBDA[0];
                solverParams.SOLVER_LAMBDA[1] = dsolve::SOLVER_LAMBDA[1];
                solverParams.SOLVER_LAMBDA[2] = dsolve::SOLVER_LAMBDA[2];
                solverParams.SOLVER_LAMBDA[3] = dsolve::SOLVER_LAMBDA[3];

                solverParams.SOLVER_LAMBDA_F[0] = dsolve::SOLVER_LAMBDA_F[0];
                solverParams.SOLVER_LAMBDA_F[1] = dsolve::SOLVER_LAMBDA_F[1];

                solverParams.SOLVER_ETA_POWER[0] = dsolve::SOLVER_ETA_POWER[0];
                solverParams.SOLVER_ETA_POWER[1] = dsolve::SOLVER_ETA_POWER[1];

                solverParams.ETA_R0 = dsolve::ETA_R0;
                solverParams.ETA_CONST = dsolve::ETA_CONST;
                solverParams.ETA_DAMPING = dsolve::ETA_DAMPING;
                solverParams.ETA_DAMPING_EXP = dsolve::ETA_DAMPING_EXP;
                solverParams.KO_DISS_SIGMA = dsolve::KO_DISS_SIGMA;

                dim3 threadBlock(16, 16, 1);
                cuda::computeRHS(
                    m_uiUnzipVarRHS, (const double **)m_uiUnzipVar,
                    &(*(blkList.begin())), blkList.size(),
                    (const cuda::SOLVERComputeParams *)&solverParams,
                    threadBlock, pt_min, pt_max, 1);
#else

                for (unsigned int blk = 0; blk < blkList.size(); blk++) {
                    std::cout << "Now computing RHS for block " << blk << " of "
                              << blkList.size() << std::endl;
                    offset = blkList[blk].getOffset();
                    sz[0] = blkList[blk].getAllocationSzX();
                    sz[1] = blkList[blk].getAllocationSzY();
                    sz[2] = blkList[blk].getAllocationSzZ();

                    bflag = blkList[blk].getBlkNodeFlag();

                    dx = blkList[blk].computeDx(pt_min, pt_max);
                    dy = blkList[blk].computeDy(pt_min, pt_max);
                    dz = blkList[blk].computeDz(pt_min, pt_max);

                    ptmin[0] = GRIDX_TO_X(blkList[blk].getBlockNode().minX()) -
                               PW * dx;
                    ptmin[1] = GRIDY_TO_Y(blkList[blk].getBlockNode().minY()) -
                               PW * dy;
                    ptmin[2] = GRIDZ_TO_Z(blkList[blk].getBlockNode().minZ()) -
                               PW * dz;

                    ptmax[0] = GRIDX_TO_X(blkList[blk].getBlockNode().maxX()) +
                               PW * dx;
                    ptmax[1] = GRIDY_TO_Y(blkList[blk].getBlockNode().maxY()) +
                               PW * dy;
                    ptmax[2] = GRIDZ_TO_Z(blkList[blk].getBlockNode().maxZ()) +
                               PW * dz;

#ifdef SOLVER_RHS_STAGED_COMP

                    solverrhs_sep(m_uiUnzipVarRHS,
                                  (const double **)m_uiUnzipVar, offset, ptmin,
                                  ptmax, sz, bflag);
#else
                    solverrhs(m_uiUnzipVarRHS, (const double **)m_uiUnzipVar,
                              offset, ptmin, ptmax, sz, bflag);
#endif
                }
#endif

#ifdef DEBUG_RK_SOLVER
                if (!rank)
                    std::cout << " stage: " << stage
                              << " af rhs UNZIP RHS TEST:" << std::endl;
                for (unsigned int index = 0; index < dsolve::SOLVER_NUM_VARS;
                     index++)
                    ot::test::isUnzipInternalNaN(m_uiMesh,
                                                 m_uiUnzipVarRHS[index]);
#endif

                zipVars(m_uiUnzipVarRHS, m_uiStage[stage]);

#ifdef DEBUG_RK_SOLVER
                for (unsigned int index = 0; index < dsolve::SOLVER_NUM_VARS;
                     index++)
                    if (seq::test::isNAN(m_uiStage[stage][index] +
                                             m_uiMesh->getNodeLocalBegin(),
                                         m_uiMesh->getNumLocalMeshNodes()))
                        std::cout << " var: " << index
                                  << " contains nan af zip  stage: " << stage
                                  << std::endl;
#endif

                /*for(unsigned int
            index=0;index<dsolve::SOLVER_NUM_VARS;index++) for(unsigned int
            node=nodeLocalBegin;node<nodeLocalEnd;node++)
                    m_uiStage[stage][index][node]*=m_uiT_h;*/

                for (unsigned int node = nodeLocalBegin; node < nodeLocalEnd;
                     node++) {
                    for (unsigned int index = 0;
                         index < dsolve::SOLVER_NUM_VARS; index++) {
                        m_uiVarIm[index][node] = m_uiPrevVar[index][node];
                        for (unsigned int s = 0; s < (stage + 1); s++) {
                            // if(!rank && index==0 && node==0)
                            // std::cout<<"rk stage: "<<stage<<" im
                            // coef:
                            // "<<s<<" value:
                            // "<<RK_U[stage+1][s]<<std::endl;
                            m_uiVarIm[index][node] +=
                                (RK_U[stage + 1][s] * m_uiT_h *
                                 m_uiStage[s][index][node]);
                        }
                    }

                    // if ( node == 101 ) {
                    // std::cout << "yo 7 node=" << node << std::endl ;
                    //}

                    enforce_system_constraints(m_uiVarIm, node);
                }

                current_t_adv = current_t + RK_T[stage + 1] * m_uiT_h;
#ifdef RK_SOLVER_OVERLAP_COMM_AND_COMP
                unzipVars_async(m_uiVarIm, m_uiUnzipVar);
#else
                performGhostExchangeVars(m_uiVarIm);
                unzipVars(m_uiVarIm, m_uiUnzipVar);
#endif
            }

            current_t_adv = current_t + RK_T[(dsolve::SOLVER_RK45_STAGES - 1)];

#ifdef DEBUG_RK_SOLVER
            if (!rank)
                std::cout << " stage: " << (dsolve::SOLVER_RK45_STAGES - 1)
                          << " begin: " << std::endl;

            for (unsigned int index = 0; index < dsolve::SOLVER_NUM_VARS;
                 index++)
                ot::test::isUnzipNaN(m_uiMesh, m_uiUnzipVar[index]);
#endif

#ifdef SOLVER_ENABLE_CUDA
            cuda::SOLVERComputeParams solverParams;
            solverParams.SOLVER_LAMBDA[0] = dsolve::SOLVER_LAMBDA[0];
            solverParams.SOLVER_LAMBDA[1] = dsolve::SOLVER_LAMBDA[1];
            solverParams.SOLVER_LAMBDA[2] = dsolve::SOLVER_LAMBDA[2];
            solverParams.SOLVER_LAMBDA[3] = dsolve::SOLVER_LAMBDA[3];

            solverParams.SOLVER_LAMBDA_F[0] = dsolve::SOLVER_LAMBDA_F[0];
            solverParams.SOLVER_LAMBDA_F[1] = dsolve::SOLVER_LAMBDA_F[1];

            solverParams.SOLVER_ETA_POWER[0] = dsolve::SOLVER_ETA_POWER[0];
            solverParams.SOLVER_ETA_POWER[1] = dsolve::SOLVER_ETA_POWER[1];

            solverParams.ETA_R0 = dsolve::ETA_R0;
            solverParams.ETA_CONST = dsolve::ETA_CONST;
            solverParams.ETA_DAMPING = dsolve::ETA_DAMPING;
            solverParams.ETA_DAMPING_EXP = dsolve::ETA_DAMPING_EXP;
            solverParams.KO_DISS_SIGMA = dsolve::KO_DISS_SIGMA;

            dim3 threadBlock(16, 16, 1);
            cuda::computeRHS(m_uiUnzipVarRHS, (const double **)m_uiUnzipVar,
                             &(*(blkList.begin())), blkList.size(),
                             (const cuda::SOLVERComputeParams *)&solverParams,
                             threadBlock, pt_min, pt_max, 1);
#else

            for (unsigned int blk = 0; blk < blkList.size(); blk++) {
                offset = blkList[blk].getOffset();
                sz[0] = blkList[blk].getAllocationSzX();
                sz[1] = blkList[blk].getAllocationSzY();
                sz[2] = blkList[blk].getAllocationSzZ();

                bflag = blkList[blk].getBlkNodeFlag();

                dx = blkList[blk].computeDx(pt_min, pt_max);
                dy = blkList[blk].computeDy(pt_min, pt_max);
                dz = blkList[blk].computeDz(pt_min, pt_max);

                ptmin[0] =
                    GRIDX_TO_X(blkList[blk].getBlockNode().minX()) - PW * dx;
                ptmin[1] =
                    GRIDY_TO_Y(blkList[blk].getBlockNode().minY()) - PW * dy;
                ptmin[2] =
                    GRIDZ_TO_Z(blkList[blk].getBlockNode().minZ()) - PW * dz;

                ptmax[0] =
                    GRIDX_TO_X(blkList[blk].getBlockNode().maxX()) + PW * dx;
                ptmax[1] =
                    GRIDY_TO_Y(blkList[blk].getBlockNode().maxY()) + PW * dy;
                ptmax[2] =
                    GRIDZ_TO_Z(blkList[blk].getBlockNode().maxZ()) + PW * dz;

#ifdef SOLVER_RHS_STAGED_COMP

                solverrhs_sep(m_uiUnzipVarRHS, (const double **)m_uiUnzipVar,
                              offset, ptmin, ptmax, sz, bflag);
#else
                solverrhs(m_uiUnzipVarRHS, (const double **)m_uiUnzipVar,
                          offset, ptmin, ptmax, sz, bflag);
#endif
            }

#endif

#ifdef DEBUG_RK_SOLVER
            if (!rank)
                std::cout << " stage: " << (dsolve::SOLVER_RK45_STAGES - 1)
                          << " af rhs UNZIP RHS TEST:" << std::endl;

            for (unsigned int index = 0; index < dsolve::SOLVER_NUM_VARS;
                 index++)
                ot::test::isUnzipInternalNaN(m_uiMesh, m_uiUnzipVarRHS[index]);
#endif

            zipVars(m_uiUnzipVarRHS,
                    m_uiStage[(dsolve::SOLVER_RK45_STAGES - 1)]);

            /*for(unsigned int
        index=0;index<dsolve::SOLVER_NUM_VARS;index++) for(unsigned int
        node=nodeLocalBegin;node<nodeLocalEnd;node++)
            m_uiStage[(dsolve::SOLVER_RK45_STAGES-1)][index][node]*=m_uiT_h;*/

            for (unsigned int node = nodeLocalBegin; node < nodeLocalEnd;
                 node++) {
                for (unsigned int index = 0; index < dsolve::SOLVER_NUM_VARS;
                     index++) {
                    m_uiVarIm[index][node] = m_uiPrevVar[index][node];
                    for (unsigned int s = 0;
                         s < (dsolve::SOLVER_RK45_STAGES - 1); s++) {
                        if (s == 1) continue;
                        m_uiVarIm[index][node] +=
                            (RK_4_C[s] * m_uiT_h * m_uiStage[s][index][node]);
                    }

                    m_uiVar[index][node] = m_uiPrevVar[index][node];
                    for (unsigned int s = 0; s < dsolve::SOLVER_RK45_STAGES;
                         s++) {
                        if (s == 1) continue;  // because rk coef is zero.
                        m_uiVar[index][node] +=
                            (RK_5_C[s] * m_uiT_h * m_uiStage[s][index][node]);
                    }
                }

                // if ( node == 101 ) {
                // std::cout << "yo 8 node=" << node << std::endl ;
                //}

                enforce_system_constraints(m_uiVarIm, node);
                enforce_system_constraints(m_uiVar, node);
            }

            // update the m_uiTh bases on the normed diff between
            // m_uiVarIm, m_uiVar.

            double n_inf;
            n_inf_max = 0;
            for (unsigned int index = 0; index < dsolve::SOLVER_NUM_VARS;
                 index++) {
                n_inf = normLInfty(m_uiVarIm[index] + nodeLocalBegin,
                                   m_uiVar[index] + nodeLocalBegin,
                                   (nodeLocalEnd - nodeLocalBegin),
                                   m_uiMesh->getMPICommunicator());
                if (n_inf > n_inf_max) n_inf_max = n_inf;
            }
        }

        // below all reduction is act as an barrier for inactive procs.
        par::Mpi_Allreduce(&n_inf_max, &n_inf_max_g, 1, MPI_MAX, m_uiComm);
        n_inf_max = n_inf_max_g;

        if (n_inf_max > dsolve::SOLVER_RK45_DESIRED_TOL) {
            repeatStep = true;
            m_uiT_h =
                dsolve::SOLVER_SAFETY_FAC * m_uiT_h *
                (pow(fabs(dsolve::SOLVER_RK45_DESIRED_TOL / n_inf_max), 0.25));
            if (!m_uiMesh->getMPIRankGlobal())
                std::cout << " repeat : " << m_uiCurrentStep
                          << " with : " << m_uiT_h << std::endl;
        } else {
            repeatStep = false;
            m_uiT_h =
                dsolve::SOLVER_SAFETY_FAC * m_uiT_h *
                (pow(fabs(dsolve::SOLVER_RK45_DESIRED_TOL / n_inf_max), 0.20));
        }

    } while (repeatStep);
}

void RK_SOLVER::performSingleIteration() {
    if (m_uiMesh->isActive()) {
        if (m_uiRKType == RKType::RK3) {
            // rk3 solver
            performSingleIterationRK3();
        } else if (m_uiRKType == RKType::RK4) {
            // rk4 solver
            performSingleIterationRK4();
        } else if (m_uiRKType == RKType::RK45) {
            // rk45 solver
            performSingleIterationRK45();
        }
    }

    m_uiMesh->waitAll();

    m_uiCurrentStep++;
    m_uiCurrentTime += m_uiT_h;
}

void RK_SOLVER::rkSolve() {
    if (m_uiCurrentStep == 0) {
        // applyInitialConditions(m_uiPrevVar);
        initialGridConverge();
    }

    bool isRefine = true;
    unsigned int oldElements, oldElements_g;
    unsigned int newElements, newElements_g;

    // refine based on all the variables
    const unsigned int refineNumVars = dsolve::SOLVER_NUM_REFINE_VARS;
    unsigned int refineVarIds[refineNumVars];
    for (unsigned int vIndex = 0; vIndex < refineNumVars; vIndex++)
        refineVarIds[vIndex] = dsolve::SOLVER_REFINE_VARIABLE_INDICES[vIndex];

    double wTol = dsolve::SOLVER_WAVELET_TOL;
    std::function<double(double, double, double, double *)> waveletTolFunc =
        [](double x, double y, double z, double *hx) {
            return dsolve::computeWTolDCoords(x, y, z, hx);
        };
    Point bhLoc[2];

    const unsigned int PW = dsolve::SOLVER_PADDING_WIDTH;
    double l_min, l_max;
    for (double t = m_uiCurrentTime; t < m_uiTimeEnd; t = t + m_uiT_h) {
        dsolve::SOLVER_CURRENT_RK_COORD_TIME = m_uiCurrentTime;
        dsolve::SOLVER_CURRENT_RK_STEP = m_uiCurrentStep;

        // checkpoint the previous solution value before going to the next step.
        dsolve::timer::t_ioCheckPoint.start();
        if ((m_uiMesh->isActive()) &&
            (m_uiCurrentStep % dsolve::SOLVER_CHECKPT_FREQ) == 0)
            storeCheckPoint(dsolve::SOLVER_CHKPT_FILE_PREFIX.c_str());
        dsolve::timer::t_ioCheckPoint.stop();

        // spit out to the console information about the current step
        if ((m_uiMesh->isActive()) &&
            (10 * m_uiCurrentStep % dsolve::SOLVER_TIME_STEP_OUTPUT_FREQ) ==
                0) {
            // TODO: could generate calculating and printting the l_min and
            // l_max for literally all of the variables

            if (!m_uiMesh->getMPIRank()) {
                std::cout << "executing step: " << m_uiCurrentStep
                          << " dt: " << m_uiT_h
                          << " rk_time : " << m_uiCurrentTime
                          << " nodes: " << m_uiMesh->getNumLocalMeshNodes()
                          << std::endl;
            }

            for (const auto tmp_varID : dsolve::SOLVER_VAR_ITERABLE_LIST) {
                l_min = vecMin(
                    m_uiPrevVar[tmp_varID] + m_uiMesh->getNodeLocalBegin(),
                    (m_uiMesh->getNumLocalMeshNodes()),
                    m_uiMesh->getMPICommunicator());
                l_max = vecMax(
                    m_uiPrevVar[tmp_varID] + m_uiMesh->getNodeLocalBegin(),
                    (m_uiMesh->getNumLocalMeshNodes()),
                    m_uiMesh->getMPICommunicator());
                if (!(m_uiMesh->getMPIRank())) {
                    std::cout
                        << "    ||VAR::" << dsolve::SOLVER_VAR_NAMES[tmp_varID]
                        << "|| (min, max) : (" << l_min << ", " << l_max
                        << " ) " << std::endl;
                }
            }

            dsolve::timer::profileInfoIntermediate(
                dsolve::SOLVER_PROFILE_FILE_PREFIX.c_str(), m_uiMesh,
                m_uiCurrentStep);
        }

        if ((m_uiCurrentStep % dsolve::SOLVER_TIME_STEP_OUTPUT_FREQ) == 0)
            dsolve::timer::resetSnapshot();

        // begin test for remeshing
        if ((m_uiCurrentStep % dsolve::SOLVER_REMESH_TEST_FREQ) == 0) {
#ifdef RK_SOLVER_OVERLAP_COMM_AND_COMP
            unzipVars_async(m_uiPrevVar, m_uiUnzipVar);
#else
            performGhostExchangeVars(m_uiPrevVar);
            // isRefine=m_uiMesh->isReMesh((const double
            // **)m_uiPrevVar,refineVarIds,refineNumVars,dsolve::SOLVER_WAVELET_TOL);
            unzipVars(m_uiPrevVar, m_uiUnzipVar);
#endif

#ifdef DEBUG_RK_SOLVER
            if (m_uiMesh->isActive()) {
                if (!m_uiMesh->getMPIRank())
                    std::cout << " isRemesh Unzip : " << std::endl;
                for (unsigned int index = 0; index < dsolve::SOLVER_NUM_VARS;
                     index++)
                    ot::test::isUnzipNaN(m_uiMesh, m_uiUnzipVar[index]);
            }
#endif
            dsolve::timer::t_isReMesh.start();
            // const double r[3] ={3.0,3.0,3.0};
            if (dsolve::SOLVER_ENABLE_BLOCK_ADAPTIVITY)
                isRefine = false;
            else {
                // TODO: some kind of edit to the refinement modes that doesn't
                // rely on black hole positioning
                if (dsolve::SOLVER_REFINEMENT_MODE ==
                    dsolve::RefinementMode::WAMR) {
                    isRefine = m_uiMesh->isReMeshUnzip(
                        (const double **)m_uiUnzipVar, refineVarIds,
                        refineNumVars, waveletTolFunc,
                        dsolve::SOLVER_DENDRO_AMR_FAC);
                    // isRefine = dsolve::isReMeshWAMR(
                    //     m_uiMesh, (const double **)m_uiUnzipVar,
                    //     refineVarIds, refineNumVars, waveletTolFunc,
                    //     dsolve::SOLVER_DENDRO_AMR_FAC);
                } else {
                    std::cout << " Error : " << __func__
                              << " invalid refinement mode specified "
                              << std::endl;
                    MPI_Abort(m_uiComm, 0);
                }
            }

            dsolve::timer::t_isReMesh.stop();

            if (isRefine) {
#ifdef DEBUG_IS_REMESH
                unsigned int rank = m_uiMesh->getMPIRankGlobal();
                MPI_Comm globalComm = m_uiMesh->getMPIGlobalCommunicator();
                std::vector<ot::TreeNode> unChanged;
                std::vector<ot::TreeNode> refined;
                std::vector<ot::TreeNode> coarsened;
                std::vector<ot::TreeNode> localBlocks;

                const ot::Block *blkList =
                    &(*(m_uiMesh->getLocalBlockList().begin()));
                for (unsigned int ele = 0;
                     ele < m_uiMesh->getLocalBlockList().size(); ele++) {
                    localBlocks.push_back(blkList[ele].getBlockNode());
                }

                const ot::TreeNode *pNodes =
                    &(*(m_uiMesh->getAllElements().begin()));
                for (unsigned int ele = m_uiMesh->getElementLocalBegin();
                     ele < m_uiMesh->getElementLocalEnd(); ele++) {
                    if ((pNodes[ele].getFlag() >> NUM_LEVEL_BITS) ==
                        OCT_NO_CHANGE) {
                        unChanged.push_back(pNodes[ele]);
                    } else if ((pNodes[ele].getFlag() >> NUM_LEVEL_BITS) ==
                               OCT_SPLIT) {
                        refined.push_back(pNodes[ele]);
                    } else {
                        assert((pNodes[ele].getFlag() >> NUM_LEVEL_BITS) ==
                               OCT_COARSE);
                        coarsened.push_back(pNodes[ele]);
                    }
                }

                char fN1[256];
                char fN2[256];
                char fN3[256];
                char fN4[256];

                sprintf(fN1, "unchanged_%d", m_uiCurrentStep);
                sprintf(fN2, "refined_%d", m_uiCurrentStep);
                sprintf(fN3, "coarsend_%d", m_uiCurrentStep);
                sprintf(fN4, "blocks_%d", m_uiCurrentStep);

                DendroIntL localSz = unChanged.size();
                DendroIntL globalSz;
                par::Mpi_Reduce(&localSz, &globalSz, 1, MPI_SUM, 0, globalComm);
                if (!rank)
                    std::cout << " total unchanged: " << globalSz << std::endl;

                localSz = refined.size();
                par::Mpi_Reduce(&localSz, &globalSz, 1, MPI_SUM, 0, globalComm);
                if (!rank)
                    std::cout << " total refined: " << globalSz << std::endl;

                localSz = coarsened.size();
                par::Mpi_Reduce(&localSz, &globalSz, 1, MPI_SUM, 0, globalComm);
                if (!rank)
                    std::cout << " total coarsend: " << globalSz << std::endl;

                io::vtk::oct2vtu(&(*(unChanged.begin())), unChanged.size(), fN1,
                                 globalComm);
                io::vtk::oct2vtu(&(*(refined.begin())), refined.size(), fN2,
                                 globalComm);
                io::vtk::oct2vtu(&(*(coarsened.begin())), coarsened.size(), fN3,
                                 globalComm);
                io::vtk::oct2vtu(&(*(localBlocks.begin())), localBlocks.size(),
                                 fN4, globalComm);

#endif
                dsolve::timer::t_mesh.start();
                ot::Mesh *newMesh = m_uiMesh->ReMesh(
                    dsolve::SOLVER_DENDRO_GRAIN_SZ, dsolve::SOLVER_LOAD_IMB_TOL,
                    dsolve::SOLVER_SPLIT_FIX);
                dsolve::timer::t_mesh.stop();

                oldElements = m_uiMesh->getNumLocalMeshElements();
                newElements = newMesh->getNumLocalMeshElements();

                par::Mpi_Reduce(&oldElements, &oldElements_g, 1, MPI_SUM, 0,
                                m_uiMesh->getMPIGlobalCommunicator());
                par::Mpi_Reduce(&newElements, &newElements_g, 1, MPI_SUM, 0,
                                newMesh->getMPIGlobalCommunicator());

                if (!(m_uiMesh->getMPIRankGlobal()))
                    std::cout << "step : " << m_uiCurrentStep
                              << " time : " << m_uiCurrentTime
                              << " old mesh: " << oldElements_g
                              << " new mesh: " << newElements_g << std::endl;

                // performs the inter-grid transfer
                intergridTransferVars(m_uiPrevVar, newMesh);

                for (unsigned int index = 0; index < dsolve::SOLVER_NUM_VARS;
                     index++) {
                    delete[] m_uiVar[index];
                    delete[] m_uiVarIm[index];
                    delete[] m_uiUnzipVar[index];
                    delete[] m_uiUnzipVarRHS[index];

                    m_uiVar[index] = NULL;
                    m_uiVarIm[index] = NULL;
                    m_uiUnzipVar[index] = NULL;
                    m_uiUnzipVarRHS[index] = NULL;

                    m_uiVar[index] = newMesh->createVector<DendroScalar>();
                    m_uiVarIm[index] = newMesh->createVector<DendroScalar>();
                    m_uiUnzipVar[index] =
                        newMesh->createUnZippedVector<DendroScalar>();
                    m_uiUnzipVarRHS[index] =
                        newMesh->createUnZippedVector<DendroScalar>();
                }

                for (unsigned int stage = 0; stage < m_uiNumRKStages; stage++)
                    for (unsigned int index = 0;
                         index < dsolve::SOLVER_NUM_VARS; index++) {
                        delete[] m_uiStage[stage][index];
                        m_uiStage[stage][index] = NULL;
                        m_uiStage[stage][index] =
                            newMesh->createVector<DendroScalar>();
                    }

                // deallocate constraint vars allocate them for the new mesh.
                for (unsigned int index = 0;
                     index < dsolve::SOLVER_CONSTRAINT_NUM_VARS; index++) {
                    delete[] m_uiConstraintVars[index];
                    delete[] m_uiUnzipConstraintVars[index];

                    m_uiConstraintVars[index] =
                        newMesh->createVector<DendroScalar>();
                    m_uiUnzipConstraintVars[index] =
                        newMesh->createUnZippedVector<DendroScalar>();
                }

                std::swap(newMesh, m_uiMesh);
                delete newMesh;

                if (m_uiCurrentStep == 0) applyInitialConditions(m_uiPrevVar);

                // now that the mesh has been swapped, reallocate the deriv
                // workspace
                dsolve::deallocate_deriv_workspace();
                dsolve::allocate_deriv_workspace(m_uiMesh, 1);

                unsigned int lmin, lmax;
                m_uiMesh->computeMinMaxLevel(lmin, lmax);
                dsolve::SOLVER_RK45_TIME_STEP_SIZE =
                    dsolve::SOLVER_CFL_FACTOR *
                    ((dsolve::SOLVER_COMPD_MAX[0] -
                      dsolve::SOLVER_COMPD_MIN[0]) *
                     ((1u << (m_uiMaxDepth - lmax)) /
                      ((double)dsolve::SOLVER_ELE_ORDER)) /
                     ((double)(1u << (m_uiMaxDepth))));
                m_uiT_h = dsolve::SOLVER_RK45_TIME_STEP_SIZE;

#ifdef RK_SOLVER_OVERLAP_COMM_AND_COMP
                // reallocates mpi resources for the the new mesh. (this will
                // deallocate the old resources)
                reallocateMPIResources();
#endif

                if (m_uiMesh->isActive()) {
                    if (!(m_uiMesh->getMPIRank())) {
                        std::cout << "TRANSFER COMPLETED, VARIABLE SUMMARY FOR "
                                  << m_uiMesh->getNumLocalMeshNodes()
                                  << " NODES:" << std::endl;
                    }

                    for (const auto tmp_varID :
                         dsolve::SOLVER_VAR_ITERABLE_LIST) {
                        l_min = vecMin(m_uiPrevVar[tmp_varID] +
                                           m_uiMesh->getNodeLocalBegin(),
                                       (m_uiMesh->getNumLocalMeshNodes()),
                                       m_uiMesh->getMPICommunicator());
                        l_max = vecMax(m_uiPrevVar[tmp_varID] +
                                           m_uiMesh->getNodeLocalBegin(),
                                       (m_uiMesh->getNumLocalMeshNodes()),
                                       m_uiMesh->getMPICommunicator());
                        if (!(m_uiMesh->getMPIRank())) {
                            std::cout << "    ||VAR::"
                                      << dsolve::SOLVER_VAR_NAMES[tmp_varID]
                                      << "|| (min, max) : (" << l_min << ", "
                                      << l_max << " ) " << std::endl;
                        }
                    }
                }
            }
        }

        dsolve::timer::t_rkStep.start();

        performSingleIteration();

#ifdef SOLVER_EXTRACT_BH_LOCATIONS
        // dsolve::extractBHCoords((const ot::Mesh *)m_uiMesh,(const
        // DendroScalar*)m_uiVar[BHLOC::EXTRACTION_VAR_ID],BHLOC::EXTRACTION_TOL,(const
        // Point *) m_uiBHLoc,2,(Point*)bhLoc);
        dsolve::computeBHLocations((const ot::Mesh *)m_uiMesh, m_uiBHLoc, bhLoc,
                                   m_uiPrevVar, m_uiT_h);
        m_uiBHLoc[0] = bhLoc[0];
        m_uiBHLoc[1] = bhLoc[1];
        dsolve::SOLVER_BH_LOC[0] = m_uiBHLoc[0];
        dsolve::SOLVER_BH_LOC[1] = m_uiBHLoc[1];
#endif

        dsolve::timer::t_rkStep.stop();

        std::swap(m_uiVar, m_uiPrevVar);
        // dsolve::artificial_dissipation(m_uiMesh,m_uiPrevVar,dsolve::SOLVER_NUM_VARS,dsolve::SOLVER_DISSIPATION_NC,dsolve::SOLVER_DISSIPATION_S,false);
        // if(m_uiCurrentStep==1) break;
    }
}

void RK_SOLVER::storeCheckPoint(const char *fNamePrefix) {
    if (m_uiMesh->isActive()) {
        unsigned int cpIndex;
        (m_uiCurrentStep % (2 * dsolve::SOLVER_CHECKPT_FREQ) == 0)
            ? cpIndex = 0
            : cpIndex = 1;  // to support alternate file writing.
        unsigned int rank = m_uiMesh->getMPIRank();
        unsigned int npes = m_uiMesh->getMPICommSize();

        char fName[256];
        const ot::TreeNode *pNodes = &(*(m_uiMesh->getAllElements().begin() +
                                         m_uiMesh->getElementLocalBegin()));
        sprintf(fName, "%s_octree_%d_%d.oct", fNamePrefix, cpIndex, rank);
        io::checkpoint::writeOctToFile(fName, pNodes,
                                       m_uiMesh->getNumLocalMeshElements());

        unsigned int numVars = dsolve::SOLVER_NUM_VARS;
        const char **varNames = dsolve::SOLVER_VAR_NAMES;

        /*for(unsigned int i=0;i<numVars;i++)
{
    sprintf(fName,"%s_%s_%d_%d.var",fNamePrefix,varNames[i],cpIndex,rank);
    io::checkpoint::writeVecToFile(fName,m_uiMesh,m_uiPrevVar[i]);
}*/

        sprintf(fName, "%s_%d_%d.var", fNamePrefix, cpIndex, rank);
        io::checkpoint::writeVecToFile(fName, m_uiMesh,
                                       (const double **)m_uiPrevVar,
                                       dsolve::SOLVER_NUM_VARS);

        if (!rank) {
            sprintf(fName, "%s_step_%d.cp", fNamePrefix, cpIndex);
            std::cout << "writing : " << fName << std::endl;
            std::ofstream outfile(fName);
            if (!outfile) {
                std::cout << fName << " file open failed " << std::endl;
                return;
            }

            json checkPoint;
            checkPoint["DENDRO_RK45_TIME_BEGIN"] = m_uiTimeBegin;
            checkPoint["DENDRO_RK45_TIME_END"] = m_uiTimeEnd;
            checkPoint["DENDRO_RK45_ELEMENT_ORDER"] = m_uiOrder;

            checkPoint["DENDRO_RK45_TIME_CURRENT"] = m_uiCurrentTime;
            checkPoint["DENDRO_RK45_STEP_CURRENT"] = m_uiCurrentStep;
            checkPoint["DENDRO_RK45_TIME_STEP_SIZE"] = m_uiT_h;
            checkPoint["DENDRO_RK45_LAST_IO_TIME"] = m_uiCurrentTime;

            checkPoint["DENDRO_RK45_WAVELET_TOLERANCE"] =
                dsolve::SOLVER_WAVELET_TOL;
            checkPoint["DENDRO_RK45_LOAD_IMB_TOLERANCE"] =
                dsolve::SOLVER_LOAD_IMB_TOL;
            checkPoint["DENDRO_RK45_NUM_VARS"] =
                numVars;  // number of variables to restore.
            checkPoint["DENDRO_RK45_ACTIVE_COMM_SZ"] =
                m_uiMesh
                    ->getMPICommSize();  // (note that rank 0 is always active).

            outfile << std::setw(4) << checkPoint << std::endl;
            outfile.close();
        }
    }
}

void RK_SOLVER::restoreCheckPoint(const char *fNamePrefix, MPI_Comm comm) {
    unsigned int numVars = 0;
    std::vector<ot::TreeNode> octree;
    json checkPoint;

    int rank;
    int npes;
    m_uiComm = comm;
    MPI_Comm_rank(m_uiComm, &rank);
    MPI_Comm_size(m_uiComm, &npes);

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
            sprintf(fName, "%s_step_%d.cp", fNamePrefix, cpIndex);
            std::ifstream infile(fName);
            if (!infile) {
                std::cout << fName << " file open failed " << std::endl;
                restoreStatus = 1;
            }

            if (restoreStatus == 0) {
                infile >> checkPoint;
                m_uiTimeBegin = checkPoint["DENDRO_RK45_TIME_BEGIN"];
                m_uiTimeEnd = checkPoint["DENDRO_RK45_TIME_END"];

                m_uiOrder = checkPoint["DENDRO_RK45_ELEMENT_ORDER"];
                m_uiCurrentTime = checkPoint["DENDRO_RK45_TIME_CURRENT"];
                m_uiCurrentStep = checkPoint["DENDRO_RK45_STEP_CURRENT"];
                m_uiT_h = checkPoint["DENDRO_RK45_TIME_STEP_SIZE"];

                dsolve::SOLVER_WAVELET_TOL =
                    checkPoint["DENDRO_RK45_WAVELET_TOLERANCE"];
                dsolve::SOLVER_LOAD_IMB_TOL =
                    checkPoint["DENDRO_RK45_LOAD_IMB_TOLERANCE"];
                numVars = checkPoint["DENDRO_RK45_NUM_VARS"];
                activeCommSz = checkPoint["DENDRO_RK45_ACTIVE_COMM_SZ"];

                restoreStep[cpIndex] = m_uiCurrentStep;
            }
        }
    }

    if (!rank) {
        if (restoreStep[0] < restoreStep[1])
            restoreFileIndex = 1;
        else
            restoreFileIndex = 0;
    }

    par::Mpi_Bcast(&restoreFileIndex, 1, 0, m_uiComm);

    for (unsigned int cpIndex = restoreFileIndex; cpIndex < 2; cpIndex++) {
        restoreStatus = 0;
        octree.clear();
        if (!rank)
            std::cout << " Trying to restore from checkpoint index : "
                      << cpIndex << std::endl;

        if (!rank) {
            sprintf(fName, "%s_step_%d.cp", fNamePrefix, cpIndex);
            std::ifstream infile(fName);
            if (!infile) {
                std::cout << fName << " file open failed " << std::endl;
                restoreStatus = 1;
            }

            if (restoreStatus == 0) {
                infile >> checkPoint;
                m_uiTimeBegin = checkPoint["DENDRO_RK45_TIME_BEGIN"];
                m_uiTimeEnd = checkPoint["DENDRO_RK45_TIME_END"];

                m_uiOrder = checkPoint["DENDRO_RK45_ELEMENT_ORDER"];
                m_uiCurrentTime = checkPoint["DENDRO_RK45_TIME_CURRENT"];
                m_uiCurrentStep = checkPoint["DENDRO_RK45_STEP_CURRENT"];
                m_uiT_h = checkPoint["DENDRO_RK45_TIME_STEP_SIZE"];

                dsolve::SOLVER_WAVELET_TOL =
                    checkPoint["DENDRO_RK45_WAVELET_TOLERANCE"];
                dsolve::SOLVER_LOAD_IMB_TOL =
                    checkPoint["DENDRO_RK45_LOAD_IMB_TOLERANCE"];
                numVars = checkPoint["DENDRO_RK45_NUM_VARS"];
                activeCommSz = checkPoint["DENDRO_RK45_ACTIVE_COMM_SZ"];
            }
        }

        par::Mpi_Allreduce(&restoreStatus, &restoreStatusGlobal, 1, MPI_MAX,
                           m_uiComm);
        if (restoreStatusGlobal == 1) continue;

        par::Mpi_Bcast(&m_uiTimeBegin, 1, 0, comm);
        par::Mpi_Bcast(&m_uiTimeEnd, 1, 0, comm);

        par::Mpi_Bcast(&m_uiCurrentTime, 1, 0, comm);
        par::Mpi_Bcast(&m_uiCurrentStep, 1, 0, comm);

        par::Mpi_Bcast(&m_uiT_h, 1, 0, comm);

        par::Mpi_Bcast(&dsolve::SOLVER_WAVELET_TOL, 1, 0, comm);
        par::Mpi_Bcast(&dsolve::SOLVER_LOAD_IMB_TOL, 1, 0, comm);

        par::Mpi_Bcast(&numVars, 1, 0, comm);
        par::Mpi_Bcast(&m_uiOrder, 1, 0, comm);
        par::Mpi_Bcast(&m_uiT_h, 1, 0, comm);

        par::Mpi_Bcast(&activeCommSz, 1, 0, comm);

        if (activeCommSz > npes) {
            if (!rank)
                std::cout << " [Error] : checkpoint file written from  a "
                             "larger communicator than the current global "
                             "comm. (i.e. communicator shrinking not allowed "
                             "in the restore step. )"
                          << std::endl;
            exit(0);
        }

        bool isActive = (rank < activeCommSz);

        MPI_Comm newComm;
        par::splitComm2way(isActive, &newComm, m_uiComm);

        if (isActive) {
            int activeRank;
            int activeNpes;

            MPI_Comm_rank(newComm, &activeRank);
            MPI_Comm_size(newComm, &activeNpes);
            assert(activeNpes == activeCommSz);

            sprintf(fName, "%s_octree_%d_%d.oct", fNamePrefix, cpIndex,
                    activeRank);
            restoreStatus = io::checkpoint::readOctFromFile(fName, octree);
            assert(par::test::isUniqueAndSorted(octree, newComm));
        }

        par::Mpi_Allreduce(&restoreStatus, &restoreStatusGlobal, 1, MPI_MAX,
                           m_uiComm);
        if (restoreStatusGlobal == 1) {
            if (!rank)
                std::cout << "[Error]: octree (*.oct) restore file currupted "
                          << std::endl;
            continue;
        }

        newMesh = new ot::Mesh(octree, 1, m_uiOrder, activeCommSz, m_uiComm);
        newMesh->setDomainBounds(
            Point(dsolve::SOLVER_GRID_MIN_X, dsolve::SOLVER_GRID_MIN_Y,
                  dsolve::SOLVER_GRID_MIN_Z),
            Point(dsolve::SOLVER_GRID_MAX_X, dsolve::SOLVER_GRID_MAX_Y,
                  dsolve::SOLVER_GRID_MAX_Z));

        for (unsigned int index = 0; index < dsolve::SOLVER_NUM_VARS; index++) {
            delete[] m_uiPrevVar[index];
            delete[] m_uiVar[index];
            delete[] m_uiVarIm[index];
            delete[] m_uiUnzipVar[index];
            delete[] m_uiUnzipVarRHS[index];

            m_uiPrevVar[index] = newMesh->createVector<DendroScalar>();
            m_uiVar[index] = newMesh->createVector<DendroScalar>();
            m_uiVarIm[index] = newMesh->createVector<DendroScalar>();
            m_uiUnzipVar[index] = newMesh->createUnZippedVector<DendroScalar>();
            m_uiUnzipVarRHS[index] =
                newMesh->createUnZippedVector<DendroScalar>();
        }

        for (unsigned int stage = 0; stage < m_uiNumRKStages; stage++)
            for (unsigned int index = 0; index < dsolve::SOLVER_NUM_VARS;
                 index++) {
                delete[] m_uiStage[stage][index];
                m_uiStage[stage][index] = newMesh->createVector<DendroScalar>();
            }

        // deallocate constraint vars allocate them for the new mesh.
        for (unsigned int index = 0; index < dsolve::SOLVER_CONSTRAINT_NUM_VARS;
             index++) {
            delete[] m_uiConstraintVars[index];
            delete[] m_uiUnzipConstraintVars[index];

            m_uiConstraintVars[index] = newMesh->createVector<DendroScalar>();
            m_uiUnzipConstraintVars[index] =
                newMesh->createUnZippedVector<DendroScalar>();
        }

        const char **varNames = dsolve::SOLVER_VAR_NAMES;

        if (isActive) {
            int activeRank;
            int activeNpes;

            MPI_Comm_rank(newComm, &activeRank);
            MPI_Comm_size(newComm, &activeNpes);
            assert(activeNpes == activeCommSz);

            /*for(unsigned int i=0;i<numVars;i++)
    {
        sprintf(fName,"%s_%s_%d_%d.var",fNamePrefix,varNames[i],cpIndex,activeRank);
        restoreStatus=io::checkpoint::readVecFromFile(fName,newMesh,m_uiPrevVar[i]);
        if(restoreStatus==1) break;

    }*/

            sprintf(fName, "%s_%d_%d.var", fNamePrefix, cpIndex, activeRank);
            restoreStatus = io::checkpoint::readVecFromFile(
                fName, newMesh, m_uiPrevVar, dsolve::SOLVER_NUM_VARS);
        }
        MPI_Comm_free(&newComm);
        par::Mpi_Allreduce(&restoreStatus, &restoreStatusGlobal, 1, MPI_MAX,
                           m_uiComm);
        if (restoreStatusGlobal == 1) {
            if (!rank)
                std::cout << "[Error]: varible (*.var) restore file currupted "
                          << std::endl;
            continue;
        }

        std::swap(m_uiMesh, newMesh);
        delete newMesh;

        dsolve::deallocate_deriv_workspace();
        dsolve::allocate_deriv_workspace(m_uiMesh, 1);

        reallocateMPIResources();
        if (restoreStatusGlobal == 0) break;
    }

    if (restoreStatusGlobal == 1) {
        std::cout << "rank: " << rank << "[Error]: rk solver restore error "
                  << std::endl;
        exit(0);
    }

    unsigned int localSz = m_uiMesh->getNumLocalMeshElements();
    unsigned int totalElems;

    par::Mpi_Allreduce(&localSz, &totalElems, 1, MPI_SUM, m_uiComm);

    if (!rank)
        std::cout << " checkpoint at step : " << m_uiCurrentStep
                  << "active Comm. sz: " << activeCommSz
                  << " restore successful: "
                  << " restored mesh size: " << totalElems << std::endl;
    return;
}

}  // namespace solver

}  // end of namespace ode
