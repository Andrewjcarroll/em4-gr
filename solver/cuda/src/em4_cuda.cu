#include <iostream>
#include <vector>

#include "EM4CtxGPU.cuh"
#include "TreeNode.h"
#include "assert.h"
#include "ets.h"
#include "gpu_profile_params.cuh"
#include "grUtils.h"
#include "mesh.h"
#include "mpi.h"
#include "octUtils.h"
#include "parameters.h"
#include "solver_main.h"
int main(int argc, char** argv) {
    unsigned int ts_mode = 1;

    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " paramFile" << std::endl;
        exit(0);
    }

    if (argc > 2) ts_mode = std::atoi(argv[2]);

    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, npes;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);

    int devicesCount;
    cudaGetDeviceCount(&devicesCount);
    if (!rank) printf("number of cuda devices: %d\n", devicesCount);
    cudaSetDevice(rank % devicesCount);

    if (!rank) {
        std::cout << CYN << BLD
                  << "==============\nNOW BEGINNING THE EM4 CUDA "
                     "SOLVER!\n==============\n"
                  << NRM << std::endl;
    }

    dsolve::timer::initFlops();

    // 1 . read the parameter file.
    if (!rank) std::cout << " reading parameter file :" << argv[1] << std::endl;
    dsolve::readParamFile(argv[1], comm);
    int root = std::min(1, npes - 1);
    dsolve::dumpParamFile(std::cout, root, comm);

    _InitializeHcurve(dsolve::SOLVER_DIM);
    m_uiMaxDepth = dsolve::SOLVER_MAXDEPTH;

    if (dsolve::SOLVER_NUM_VARS % dsolve::SOLVER_ASYNC_COMM_K != 0) {
        if (!rank)
            std::cout
                << "[overlap communication error]: total SOLVER_NUM_VARS: "
                << dsolve::SOLVER_NUM_VARS
                << " is not divisable by SOLVER_ASYNC_COMM_K: "
                << dsolve::SOLVER_ASYNC_COMM_K << std::endl;
        exit(0);
    }

    // 2. generate the initial grid.
    std::vector<ot::TreeNode> tmpNodes;
    std::function<void(double, double, double, double*)> f_init =
        [](double x, double y, double z, double* var) {
            dsolve::initDataFuncToPhysCoords(x, y, z, var);
        };

    const unsigned int interpVars = dsolve::SOLVER_NUM_VARS;
    unsigned int varIndex[interpVars];
    for (unsigned int i = 0; i < dsolve::SOLVER_NUM_VARS; i++) varIndex[i] = i;

    if (dsolve::SOLVER_ENABLE_BLOCK_ADAPTIVITY) {
        if (!rank)
            std::cout << YLW << "Using block adaptive mesh. AMR disabled "
                      << NRM << std::endl;
        const Point pt_min(dsolve::SOLVER_BLK_MIN_X, dsolve::SOLVER_BLK_MIN_Y,
                           dsolve::SOLVER_BLK_MIN_Z);
        const Point pt_max(dsolve::SOLVER_BLK_MAX_X, dsolve::SOLVER_BLK_MAX_Y,
                           dsolve::SOLVER_BLK_MAX_Z);

        dsolve::blockAdaptiveOctree(
            tmpNodes, pt_min, pt_max,
            m_uiMaxDepth - (binOp::fastLog2(dsolve::SOLVER_ELE_ORDER)),
            m_uiMaxDepth, comm);
    } else {
        if (!rank)
            std::cout << YLW << "Using function2Octree. AMR enabled " << NRM
                      << std::endl;
        unsigned int maxDepthIn = m_uiMaxDepth - 2;

        function2Octree(f_init, dsolve::SOLVER_NUM_VARS, varIndex, interpVars,
                        tmpNodes, maxDepthIn, dsolve::SOLVER_WAVELET_TOL,
                        dsolve::SOLVER_ELE_ORDER, comm);
    }

    if (!rank) std::cout << "Now generating mesh" << std::endl;

    ot::Mesh* mesh = ot::createMesh(
        tmpNodes.data(), tmpNodes.size(), dsolve::SOLVER_ELE_ORDER, comm, 1,
        ot::SM_TYPE::FDM, dsolve::SOLVER_DENDRO_GRAIN_SZ,
        dsolve::SOLVER_LOAD_IMB_TOL, dsolve::SOLVER_SPLIT_FIX);

    if (!rank) std::cout << "Mesh generation finished" << std::endl;

    mesh->setDomainBounds(
        Point(dsolve::SOLVER_GRID_MIN_X, dsolve::SOLVER_GRID_MIN_Y,
              dsolve::SOLVER_GRID_MIN_Z),
        Point(dsolve::SOLVER_GRID_MAX_X, dsolve::SOLVER_GRID_MAX_Y,
              dsolve::SOLVER_GRID_MAX_Z));

    unsigned int lmin, lmax;
    mesh->computeMinMaxLevel(lmin, lmax);
    dsolve::SOLVER_RK45_TIME_STEP_SIZE =
        dsolve::SOLVER_CFL_FACTOR *
        ((dsolve::SOLVER_COMPD_MAX[0] - dsolve::SOLVER_COMPD_MIN[0]) *
         ((1u << (m_uiMaxDepth - lmax)) / ((double)dsolve::SOLVER_ELE_ORDER)) /
         ((double)(1u << (m_uiMaxDepth))));
    par::Mpi_Bcast(&dsolve::SOLVER_RK45_TIME_STEP_SIZE, 1, 0, comm);

    if (!rank) {
        std::cout << "lmin: " << lmin << " lmax:" << lmax << std::endl;
        std::cout << "dt: " << dsolve::SOLVER_RK45_TIME_STEP_SIZE << std::endl;
    }

    if (ts_mode == 1) {
        cuda::EM4CtxGPU* appCtx = new cuda::EM4CtxGPU(mesh);
        ts::ETS<DendroScalar, cuda::EM4CtxGPU>* ets =
            new ts::ETS<DendroScalar, cuda::EM4CtxGPU>(appCtx);
        ets->set_evolve_vars(appCtx->get_evolution_vars());

        if ((RKType)dsolve::SOLVER_RK_TYPE == RKType::RK3)
            ets->set_ets_coefficients(ts::ETSType::RK3);
        else if ((RKType)dsolve::SOLVER_RK_TYPE == RKType::RK4)
            ets->set_ets_coefficients(ts::ETSType::RK4);
        else if ((RKType)dsolve::SOLVER_RK_TYPE == RKType::RK5)
            ets->set_ets_coefficients(ts::ETSType::RK5);

        ets->init();

        double t1 = MPI_Wtime();
        for (ets->init(); ets->curr_time() < dsolve::SOLVER_RK_TIME_END;) {
            const DendroIntL step          = ets->curr_step();
            const DendroScalar time        = ets->curr_time();

            const bool isActive            = ets->is_active();
            const unsigned int rank_global = ets->get_global_rank();

            if (dsolve::SOLVER_REMESH_TEST_FREQ > 0 &&
                (step % dsolve::SOLVER_REMESH_TEST_FREQ) == 0 && step != 0) {
                appCtx->device_to_host_sync();
                dsolve::timer::t_isReMesh.start();
                bool isRemesh = appCtx->is_remesh();
                dsolve::timer::t_isReMesh.stop();
                if (isRemesh) {
                    dsolve::timer::t_remesh.start();
                    appCtx->remesh_and_gridtransfer(
                        dsolve::SOLVER_DENDRO_GRAIN_SZ,
                        dsolve::SOLVER_LOAD_IMB_TOL, dsolve::SOLVER_SPLIT_FIX);
                    dsolve::timer::t_remesh.stop();
                    ets->sync_with_mesh();
                }
            }

            if (dsolve::SOLVER_TIME_STEP_OUTPUT_FREQ > 0 &&
                (step % dsolve::SOLVER_TIME_STEP_OUTPUT_FREQ) == 0) {
                if (!rank_global)
                    std::cout << BLD << GRN << "==========\n"
                              << "[ETS - SOLVER] : SOLVER UPDATE\n"
                              << NRM << "\tCurrent Step: " << ets->curr_step()
                              << "\t\tCurrent time: " << ets->curr_time()
                              << "\tdt: " << ets->ts_size() << std::endl;
                appCtx->terminal_output();
            }

            if (dsolve::SOLVER_IO_OUTPUT_FREQ > 0 &&
                (step % dsolve::SOLVER_IO_OUTPUT_FREQ) == 0) {
                appCtx->write_vtu();
            }

            if (dsolve::SOLVER_CHECKPT_FREQ > 0 &&
                (step % dsolve::SOLVER_CHECKPT_FREQ) == 0) {
                std::cout << "    NOW writing checkpoint" << std::endl;
                appCtx->write_checkpt();
                appCtx->get_mesh()->waitAll();
                std::cout << "    finished checkpoint" << std::endl;
            }

            appCtx->resetForEvolutionStuff();
            dsolve::timer::t_rkStep.start();
            ets->evolve();
            dsolve::timer::t_rkStep.stop();
            appCtx->resetForNextStep();
        }

        double t2 = MPI_Wtime() - t1;
        double t2_g;
        par::Mpi_Allreduce(&t2, &t2_g, 1, MPI_MAX, ets->get_global_comm());
        if (!rank) std::cout << " ETS time (max) : " << t2_g << std::endl;

        appCtx->finalize();

#ifdef EM4_ENABLE_PROFILING
        dsolve::timer::profileInfo(
            dsolve::SOLVER_PROFILE_FILE_PREFIX.c_str(),
            appCtx->get_mesh());

        dsolve::gpu_timer::profileInfoGPU(
            dsolve::SOLVER_PROFILE_FILE_PREFIX.c_str(),
            appCtx->get_mesh());
#endif

        delete appCtx->get_mesh();
        delete appCtx;
        delete ets;
    }

    MPI_Finalize();
    return 0;
}
