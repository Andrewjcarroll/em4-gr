
#include <iostream>
#include <vector>

#include "TreeNode.h"
#include "grUtils.h"
#include "mesh.h"
#include "meshUtils.h"
#include "mpi.h"
#include "octUtils.h"
#include "parameters.h"
#include "rkSolver.h"
#include "solver_main.h"

int em3_run_driver(MPI_Comm comm, unsigned int num_step, unsigned int warm_up,
                   std::ostream& outfile, unsigned int ts_mode) {
    int rank, npes;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);

    std::vector<ot::TreeNode> tmpNodes;

    std::function<void(double, double, double, double*)> f_init =
        [](double x, double y, double z, double* var) {
            dsolve::initDataFuncToPhysCoords(x, y, z, var);
        };

    const unsigned int interpVars = dsolve::SOLVER_NUM_VARS;

    unsigned int varIndex[interpVars];
    for (unsigned int i = 0; i < dsolve::SOLVER_NUM_VARS; i++) varIndex[i] = i;

    if (false && dsolve::SOLVER_ENABLE_BLOCK_ADAPTIVITY) {
        if (!rank)
            std::cout << YLW << "Using block adaptive mesh. AMR disabled "
                      << NRM << std::endl;
        const Point pt_min(dsolve::SOLVER_BLK_MIN_X, dsolve::SOLVER_BLK_MIN_Y,
                           dsolve::SOLVER_BLK_MIN_Z);
        const Point pt_max(dsolve::SOLVER_BLK_MAX_X, dsolve::SOLVER_BLK_MAX_Y,
                           dsolve::SOLVER_BLK_MAX_Z);
        dsolve::blockAdaptiveOctree(tmpNodes, pt_min, pt_max, m_uiMaxDepth - 2,
                                    m_uiMaxDepth, comm);
    } else {
        if (!rank)
            std::cout << YLW << "Using function2Octree. AMR enabled " << NRM
                      << std::endl;

        unsigned int maxDepthIn;

        maxDepthIn = dsolve::SOLVER_MAXDEPTH - 2;

        function2Octree(f_init, dsolve::SOLVER_NUM_VARS, varIndex, interpVars,
                        tmpNodes, maxDepthIn, dsolve::SOLVER_WAVELET_TOL,
                        dsolve::SOLVER_ELE_ORDER, comm);
    }

    // std::vector<ot::TreeNode> f2Octants(tmpNodes);
    ot::Mesh* mesh = ot::createMesh(
        tmpNodes.data(), tmpNodes.size(), dsolve::SOLVER_ELE_ORDER, comm, 1,
        ot::SM_TYPE::FDM, dsolve::SOLVER_DENDRO_GRAIN_SZ,
        dsolve::SOLVER_LOAD_IMB_TOL, dsolve::SOLVER_SPLIT_FIX);
    mesh->setDomainBounds(
        Point(dsolve::SOLVER_GRID_MIN_X, dsolve::SOLVER_GRID_MIN_Y,
              dsolve::SOLVER_GRID_MIN_Z),
        Point(dsolve::SOLVER_GRID_MAX_X, dsolve::SOLVER_GRID_MAX_Y,
              dsolve::SOLVER_GRID_MAX_Z));

    unsigned int lmin, lmax;
    mesh->computeMinMaxLevel(lmin, lmax);

    if (!rank) {
        std::cout << "================= Grid Info (Before init grid "
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
        std::cout << "ts mode: " << ts_mode << std::endl;
        std::cout
            << "============================================================="
               "=================================================="
            << std::endl;
    }

    dsolve::SOLVER_RK45_TIME_STEP_SIZE =
        dsolve::SOLVER_CFL_FACTOR *
        ((dsolve::SOLVER_COMPD_MAX[0] - dsolve::SOLVER_COMPD_MIN[0]) *
         ((1u << (m_uiMaxDepth - lmax)) / ((double)dsolve::SOLVER_ELE_ORDER)) /
         ((double)(1u << (m_uiMaxDepth))));
    tmpNodes.clear();

    // enable block adaptivity to disable the remesh
    dsolve::SOLVER_ENABLE_BLOCK_ADAPTIVITY = 1;
    ot::Mesh* pMesh = mesh;

    if (dsolve::SOLVER_RESTORE_SOLVER == 0)
        pMesh = dsolve::weakScalingReMesh(mesh, npes);

    if (ts_mode == 0) {
    } else if (ts_mode == 1) {
        dsolve::SOLVERCtx* solverCtx = new dsolve::SOLVERCtx(pMesh);
        ts::ETS<DendroScalar, dsolve::SOLVERCtx>* ets =
            new ts::ETS<DendroScalar, dsolve::SOLVERCtx>(solverCtx);
        ets->set_evolve_vars(solverCtx->get_evolution_vars());

        if ((RKType)dsolve::SOLVER_RK_TYPE == RKType::RK3)
            ets->set_ets_coefficients(ts::ETSType::RK3);
        else if ((RKType)dsolve::SOLVER_RK_TYPE == RKType::RK4)
            ets->set_ets_coefficients(ts::ETSType::RK4);
        else if ((RKType)dsolve::SOLVER_RK_TYPE == RKType::RK45)
            ets->set_ets_coefficients(ts::ETSType::RK5);

        for (ets->init(); ets->curr_step() < (warm_up + num_step);
             ets->evolve()) {
            const DendroIntL step = ets->curr_step();
            const DendroScalar time = ets->curr_time();
            pMesh = solverCtx->get_mesh();

#if defined __PROFILE_ETS__ && __PROFILE_CTX__
            if (step == warm_up) {
                ets->reset_pt();
                solverCtx->reset_pt();
            }
#endif

            // if( step == 0 )
            //   solverCtx -> write_checkpt();

            dsolve::SOLVER_CURRENT_RK_COORD_TIME = step;
            dsolve::SOLVER_CURRENT_RK_STEP = time;

            const bool isActive = ets->is_active();
            const unsigned int rank_global = ets->get_global_rank();

            if (!rank_global)
                std::cout << "[ETS] : Executing step :  " << ets->curr_step()
                          << "\tcurrent time :" << ets->curr_time()
                          << "\t dt:" << ets->ts_size() << "\t" << std::endl;
        }

#if defined __PROFILE_ETS__ && __PROFILE_CTX__
        pMesh = solverCtx->get_mesh();
        if (pMesh->isActive()) {
            int active_rank, active_npes;
            MPI_Comm active_comm = pMesh->getMPICommunicator();

            MPI_Comm_rank(active_comm, &active_rank);
            MPI_Comm_size(active_comm, &active_npes);

            if (!active_rank)
                outfile
                    << "act_npes\tglb_npes\tmaxdepth\twarm_up\tnum_"
                       "steps\tnumOcts\tdof_cg\tdof_uz\t"
                    << "gele_min\tgele_mean\tgele_max\t"
                       "lele_min\tlele_mean\tlele_max\t"
                       "gnodes_min\tgnodes_mean\tgnodes_max\t"
                       "lnodes_min\tlnodes_mean\tlnodes_max\t"
                       "remsh_min\tremsh_mean\tremsh_max\t"
                       "remsh_igt_min\tremsh_igt_mean\tremsh_igt_max\t"
                       "evolve_min\tevolve_mean\tevolve_max\t"
                       "unzip_wcomm_min\tunzip_wcomm_mean\tunzip_wcomm_max\t"
                       "unzip_min\tunzip_mean\tunzip_max\t"
                       "rhs_min\trhs_mean\trhs_max\t"
                       "rhs_blk_min\trhs_blk_mean\trhs_blk_max\t"
                       "zip_wcomm_min\tzip_wcomm_mean\tzip_wcomm_max\t"
                       "zip_min\tzip_mean\tzip_max\t"
                    << std::endl;

            if (!rank) outfile << active_npes << "\t";
            if (!rank) outfile << npes << "\t";
            if (!rank) outfile << dsolve::SOLVER_MAXDEPTH << "\t";
            if (!rank) outfile << warm_up << "\t";
            if (!rank) outfile << num_step << "\t";

            DendroIntL localSz = pMesh->getNumLocalMeshElements();
            DendroIntL globalSz;

            par::Mpi_Reduce(&localSz, &globalSz, 1, MPI_SUM, 0, active_comm);
            if (!rank) outfile << globalSz << "\t";

            localSz = pMesh->getNumLocalMeshNodes();
            par::Mpi_Reduce(&localSz, &globalSz, 1, MPI_SUM, 0, active_comm);
            if (!rank) outfile << globalSz << "\t";

            localSz = pMesh->getDegOfFreedomUnZip();
            par::Mpi_Reduce(&localSz, &globalSz, 1, MPI_SUM, 0, active_comm);
            if (!rank) outfile << globalSz << "\t";

            DendroIntL ghostElements = pMesh->getNumPreGhostElements() +
                                       pMesh->getNumPostGhostElements();
            DendroIntL localElements = pMesh->getNumLocalMeshElements();

            double t_stat = ghostElements;
            double t_stat_g[3];
            min_mean_max(&t_stat, t_stat_g, active_comm);
            if (!rank)
                outfile << t_stat_g[0] << "\t" << t_stat_g[1] << "\t"
                        << t_stat_g[2] << "\t";

            t_stat = localElements;
            min_mean_max(&t_stat, t_stat_g, active_comm);
            if (!rank)
                outfile << t_stat_g[0] << "\t" << t_stat_g[1] << "\t"
                        << t_stat_g[2] << "\t";

            DendroIntL ghostNodes =
                pMesh->getNumPreMeshNodes() + pMesh->getNumPostMeshNodes();
            DendroIntL localNodes = pMesh->getNumLocalMeshNodes();

            t_stat = ghostNodes;
            min_mean_max(&t_stat, t_stat_g, active_comm);
            if (!rank)
                outfile << t_stat_g[0] << "\t" << t_stat_g[1] << "\t"
                        << t_stat_g[2] << "\t";

            t_stat = localNodes;
            min_mean_max(&t_stat, t_stat_g, active_comm);
            if (!rank)
                outfile << t_stat_g[0] << "\t" << t_stat_g[1] << "\t"
                        << t_stat_g[2] << "\t";

            t_stat = solverCtx->m_uiCtxpt[ts::CTXPROFILE::REMESH].snap;
            min_mean_max(&t_stat, t_stat_g, active_comm);
            if (!rank)
                outfile << t_stat_g[0] << "\t" << t_stat_g[1] << "\t"
                        << t_stat_g[2] << "\t";

            t_stat = solverCtx->m_uiCtxpt[ts::CTXPROFILE::GRID_TRASFER].snap;
            min_mean_max(&t_stat, t_stat_g, active_comm);
            if (!rank)
                outfile << t_stat_g[0] << "\t" << t_stat_g[1] << "\t"
                        << t_stat_g[2] << "\t";

            t_stat = ets->m_uiCtxpt[ts::ETSPROFILE::EVOLVE].snap;
            min_mean_max(&t_stat, t_stat_g, active_comm);
            if (!rank)
                outfile << t_stat_g[0] << "\t" << t_stat_g[1] << "\t"
                        << t_stat_g[2] << "\t";

            t_stat = solverCtx->m_uiCtxpt[ts::CTXPROFILE::UNZIP_WCOMM].snap;
            min_mean_max(&t_stat, t_stat_g, active_comm);
            if (!rank)
                outfile << t_stat_g[0] << "\t" << t_stat_g[1] << "\t"
                        << t_stat_g[2] << "\t";

            t_stat = solverCtx->m_uiCtxpt[ts::CTXPROFILE::UNZIP].snap;
            min_mean_max(&t_stat, t_stat_g, active_comm);
            if (!rank)
                outfile << t_stat_g[0] << "\t" << t_stat_g[1] << "\t"
                        << t_stat_g[2] << "\t";

            t_stat = solverCtx->m_uiCtxpt[ts::CTXPROFILE::RHS].snap;
            min_mean_max(&t_stat, t_stat_g, active_comm);
            if (!rank)
                outfile << t_stat_g[0] << "\t" << t_stat_g[1] << "\t"
                        << t_stat_g[2] << "\t";

            t_stat = solverCtx->m_uiCtxpt[ts::CTXPROFILE::RHS_BLK].snap;
            min_mean_max(&t_stat, t_stat_g, active_comm);
            if (!rank)
                outfile << t_stat_g[0] << "\t" << t_stat_g[1] << "\t"
                        << t_stat_g[2] << "\t";

            t_stat = solverCtx->m_uiCtxpt[ts::CTXPROFILE::ZIP_WCOMM].snap;
            min_mean_max(&t_stat, t_stat_g, active_comm);
            if (!rank)
                outfile << t_stat_g[0] << "\t" << t_stat_g[1] << "\t"
                        << t_stat_g[2] << "\t";

            t_stat = solverCtx->m_uiCtxpt[ts::CTXPROFILE::ZIP].snap;
            min_mean_max(&t_stat, t_stat_g, active_comm);
            if (!rank)
                outfile << t_stat_g[0] << "\t" << t_stat_g[1] << "\t"
                        << t_stat_g[2] << "\t";
            if (!rank) outfile << std::endl;
        }
#endif

        // ets->m_uiCtxpt
        // std::cout<<"reached end:"<<rank<<std::endl;

        ot::Mesh* tmp_mesh = solverCtx->get_mesh();
        delete solverCtx;
        delete tmp_mesh;
        delete ets;

    } else {
        if (!rank) RAISE_ERROR("invalid ts mode : " << ts_mode << "specifed");

        MPI_Abort(comm, 0);
    }

    return 0;
}

int main(int argc, char** argv) {
    unsigned int ts_mode = 1;

    if (argc < 2) {
        std::cout << "No parameter file was given, exiting..." << std::endl;
        std::cout << "Usage: " << argv[0] << " paramFile" << std::endl;
        exit(0);
    }

    // get the time stepper mode
    if (argc > 2) ts_mode = std::atoi(argv[2]);

    // seed the randomness used later
    srand(static_cast<unsigned>(time(0)));

    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, npes;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);

    // initialize the flops timer
    dsolve::timer::initFlops();

    // begin the imer for total run time
    dsolve::timer::total_runtime.start();

    // CMake options to be printed:::
    if (!rank) {
        std::cout << CYN << BLD
                  << "==============\nNOW BEGINNING THE "
                     "SOLVER!\n==============\n"
                  << NRM << std::endl;
#ifdef SOLVER_COMPUTE_CONSTRAINTS
        std::cout << GRN << "  Compiled with SOLVER_COMPUTE_CONSTRAINTS" << NRM
                  << std::endl;
#else
        std::cout << RED << "  Compiled without SOLVER_COMPUTE_CONSTRAINTS"
                  << NRM << std::endl;
#endif
#ifdef SOLVER_ENABLE_VTU_CONSTRAINT_OUTPUT
        std::cout << GRN
                  << "  Compiled with SOLVER_ENABLE_VTU_CONSTRAINT_OUTPUT"
                  << NRM << std::endl;
#else
        std::cout << RED
                  << "  Compiled without SOLVER_ENABLE_VTU_CONSTRAINT_OUTPUT"
                  << NRM << std::endl;
#endif
#ifdef SOLVER_ENABLE_VTU_OUTPUT
        std::cout << GRN << "  Compiled with SOLVER_ENABLE_VTU_OUTPUT" << NRM
                  << std::endl;
#else
        std::cout << RED << "  Compiled without SOLVER_ENABLE_VTU_OUTPUT" << NRM
                  << std::endl;
#endif

#ifdef USE_FD_INTERP_FOR_UNZIP
        std::cout << GRN << "  Compiled with  USE_FD_INTERP_FOR_UNZIP" << NRM
                  << std::endl;
#else
        std::cout << RED << "  Compiled without  USE_FD_INTERP_FOR_UNZIP" << NRM
                  << std::endl;
#endif

#ifdef SOLVER_USE_4TH_ORDER_DERIVS
        std::cout << GRN << "  Using 4th order FD stencils" << NRM << std::endl;
#endif

#ifdef SOLVER_USE_6TH_ORDER_DERIVS
        std::cout << GRN << "  Using 6th order FD stencils" << NRM << std::endl;
#endif

#ifdef SOLVER_USE_8TH_ORDER_DERIVS
        std::cout << GRN << "  Using 8th order FD stencils" << NRM << std::endl;
#endif
    }

    /**
     * STEP 1
     *
     * Read in the Parameter File and check initialization parameters for
     * problems.
     */
    if (!rank) {
        std::cout << " reading parameter file :" << argv[1] << std::endl;
    }
    dsolve::readParamFile(argv[1], comm);

    int root = std::min(1, npes - 1);
    // dump parameter file
    dsolve::dumpParamFile(std::cout, root, comm);

    _InitializeHcurve(dsolve::SOLVER_DIM);
    m_uiMaxDepth = dsolve::SOLVER_MAXDEPTH;

    if (dsolve::SOLVER_NUM_VARS % dsolve::SOLVER_ASYNC_COMM_K != 0) {
        if (!rank) {
            std::cout
                << "[overlap communication error]: total SOLVER_NUM_VARS: "
                << dsolve::SOLVER_NUM_VARS
                << " is not divisable by SOLVER_ASYNC_COMM_K: "
                << dsolve::SOLVER_ASYNC_COMM_K << std::endl;
        }
        MPI_Abort(comm, 0);
    }

    std::vector<ot::TreeNode> tmpNodes;

    std::function<void(double, double, double, double*)> f_init =
        [](double x, double y, double z, double* var) {
            dsolve::initDataFuncToPhysCoords(x, y, z, var);
        };

    const unsigned int interpVars = dsolve::SOLVER_NUM_VARS;
    unsigned int varIndex[interpVars];
    for (unsigned int i = 0; i < dsolve::SOLVER_NUM_VARS; i++) {
        varIndex[i] = i;
    }

    DendroIntL localSz, globalSz;
    double t_stat;
    double t_stat_g[3];

    const unsigned int NUM_WARM_UP = 0;
    const unsigned int NUM_STEPS = 5;

    std::ofstream outfile;
    char fname[256];
    sprintf(fname, "solverCtx_WS_%d.txt", npes);

    if (!rank) {
        outfile.open(fname, std::ios_base::app);
        time_t now = time(0);
        // convert now to string form
        char* dt = ctime(&now);
        outfile
            << "============================================================"
            << std::endl;
        outfile << "Current time : " << dt << " --- " << std::endl;
        outfile
            << "============================================================"
            << std::endl;
        outfile << " element order: " << dsolve::SOLVER_ELE_ORDER << std::endl;
        outfile << "  Compiled with: ";

#ifdef SOLVER_USE_4TH_ORDER_DERIVS
        outfile << " FOURTH Order Derivatives " << std::endl;
#endif
#ifdef SOLVER_USE_6TH_ORDER_DERIVS
        outfile << " SIXTH Order Derivatives " << std::endl;
#endif
#ifdef SOLVER_USE_8TH_ORDER_DERIVS
        outfile << " EIGHTH Order Derivatives " << std::endl;
#endif
        outfile
            << "============================================================"
            << std::endl;
    }

    em3_run_driver(comm, NUM_STEPS, NUM_WARM_UP, outfile, 1);

    if (!rank) outfile.close();

#ifdef RUN_WEAK_SCALING

    if (!rank)
        std::cout << "========================================================="
                     "============="
                  << std::endl;
    if (!rank) std::cout << "     Weak Scaling Run Begin.   " << std::endl;
    if (!rank)
        std::cout << "========================================================="
                     "============="
                  << std::endl;

    int proc_group = 0;
    int min_np = 2;
    for (int i = npes; rank < i && i >= min_np; i = i >> 1) proc_group++;
    MPI_Comm comm_ws;

    MPI_Comm_split(comm, proc_group, rank, &comm_ws);

    MPI_Comm_rank(comm_ws, &rank);
    MPI_Comm_size(comm_ws, &npes);

    if (!rank) outfile.open(fname, std::ios_base::app);
    MPI_Barrier(comm_ws);

    em3_run_driver(comm_ws, NUM_STEPS, NUM_WARM_UP, outfile, 1);

    MPI_Barrier(comm_ws);
    if (!rank) outfile.close();

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);
    MPI_Barrier(comm);

    if (!rank)
        std::cout << "========================================================="
                     "============="
                  << std::endl;
    if (!rank) std::cout << "     Weak Scaling Run Complete.   " << std::endl;
    if (!rank)
        std::cout << "========================================================="
                     "============="
                  << std::endl;

#endif

    MPI_Finalize();
    return 0;
}
