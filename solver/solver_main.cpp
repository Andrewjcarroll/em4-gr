/**
 * This is the main file that executes the solver
 *
 * @author David Van Komen, Milinda Fernando
 * @brief The main file that drives the code
 *
 * :::License:::
 */

#include "solver_main.h"

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

// dump parameter data to individual files for sanity
// uncomment the following to test the dump for each individual process!
#if 0
    for (int ifile = 0; ifile < npes; ifile++) {
        if (rank == ifile) {
            std::ofstream tempfile;
            tempfile.open("dumped_params_process_" + std::to_string(ifile) +
                          ".txt");
            dsolve::dumpParamFile(tempfile, ifile, comm);
            tempfile.close();
        }
    }
#endif

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

    /**
     * STEP 2
     *
     * Generate the initial grid
     * Either through adaptive blocking or through adaptive mesh refinement
     */
    std::vector<ot::TreeNode> tmpNodes;

    // NOTE: initial grid function needs to do a conversion from grid format to
    // physical format f_init is only used when SOLVER_ENABLE_BLOCK_ADAPTIVITY
    // is turned off
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

    dsolve::timer::t_f2o.start();

    if (dsolve::SOLVER_ENABLE_BLOCK_ADAPTIVITY) {
        if (!rank) {
            std::cout << YLW << "Using block adaptive mesh. AMR disabled "
                      << NRM << std::endl;
        }
        const Point pt_min(dsolve::SOLVER_BLK_MIN_X, dsolve::SOLVER_BLK_MIN_Y,
                           dsolve::SOLVER_BLK_MIN_Z);
        const Point pt_max(dsolve::SOLVER_BLK_MAX_X, dsolve::SOLVER_BLK_MAX_Y,
                           dsolve::SOLVER_BLK_MAX_Z);

        dsolve::blockAdaptiveOctree(tmpNodes, pt_min, pt_max, m_uiMaxDepth - 2,
                                    m_uiMaxDepth, comm);
    } else {
        if (!rank) {
            std::cout << YLW << "Using function2Octree. AMR enabled " << NRM
                      << std::endl;
        }

        // f2olmin is like the max depth we want to refine to.
        // if we don't have two puncture initial data, then it should just be
        // the max depth minus three
        unsigned int maxDepthIn;

        // max depth in to the function2Octree must be 2 less than the max depth
        maxDepthIn = m_uiMaxDepth - 2;

        function2Octree(f_init, dsolve::SOLVER_NUM_VARS, varIndex, interpVars,
                        tmpNodes, maxDepthIn, dsolve::SOLVER_WAVELET_TOL,
                        dsolve::SOLVER_ELE_ORDER, comm);
    }

    /**
     * STEP 3
     *
     * Generate the mesh itself
     * Take the initial grid and set up the mesh to be used throughout
     * computations
     */
    if (!rank) {
        std::cout << "Now generating mesh" << std::endl;
    }

    ot::Mesh* mesh = ot::createMesh(
        tmpNodes.data(), tmpNodes.size(), dsolve::SOLVER_ELE_ORDER, comm, 1,
        ot::SM_TYPE::FDM, dsolve::SOLVER_DENDRO_GRAIN_SZ,
        dsolve::SOLVER_LOAD_IMB_TOL, dsolve::SOLVER_SPLIT_FIX);

    if (!rank) {
        std::cout << "Mesh generation finished" << std::endl;
    }
    mesh->setDomainBounds(
        Point(dsolve::SOLVER_GRID_MIN_X, dsolve::SOLVER_GRID_MIN_Y,
              dsolve::SOLVER_GRID_MIN_Z),
        Point(dsolve::SOLVER_GRID_MAX_X, dsolve::SOLVER_GRID_MAX_Y,
              dsolve::SOLVER_GRID_MAX_Z));

    // io::vtk::mesh2vtuFine(mesh,"begin",0,NULL,NULL,0,NULL,NULL,0,NULL,NULL,false);

    if (!rank) {
        std::cout << "Domain bounds set" << std::endl;
    }

    unsigned int lmin, lmax;
    mesh->computeMinMaxLevel(lmin, lmax);

    if (!rank) {
        std::cout << "Computed min max level:" << lmin << " " << lmax
                  << std::endl;
    }

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

    /**
     * STEP 4
     *
     * Prepare for RK solving
     * Includes prepping the the solver and potentially restoring from a
     * checkpoint if enabled
     */
    dsolve::SOLVER_RK45_TIME_STEP_SIZE =
        dsolve::SOLVER_CFL_FACTOR *
        ((dsolve::SOLVER_COMPD_MAX[0] - dsolve::SOLVER_COMPD_MIN[0]) *
         ((1u << (m_uiMaxDepth - lmax)) / ((double)dsolve::SOLVER_ELE_ORDER)) /
         ((double)(1u << (m_uiMaxDepth))));
    tmpNodes.clear();

    if (ts_mode == 1) {
        if (!rank) {
            std::cout << CYN << BLD << "Now initializing time stepper..." << NRM
                      << std::endl;
        }
        dsolve::SOLVERCtx* solverCtx = new dsolve::SOLVERCtx(mesh);
        if (!rank) {
            std::cout << CYN << BLD << "Time stepper successfully initialized!"
                      << NRM << std::endl;
        }
        ts::ETS<DendroScalar, dsolve::SOLVERCtx>* ets =
            new ts::ETS<DendroScalar, dsolve::SOLVERCtx>(solverCtx);
        ets->set_evolve_vars(solverCtx->get_evolution_vars());
        if (!rank) {
            std::cout << CYN << BLD << "Evolution variables now set..." << NRM
                      << std::endl;
        }

        if ((RKType)dsolve::SOLVER_RK_TYPE == RKType::RK3)
            ets->set_ets_coefficients(ts::ETSType::RK3);
        else if ((RKType)dsolve::SOLVER_RK_TYPE == RKType::RK4)
            ets->set_ets_coefficients(ts::ETSType::RK4);
        else if ((RKType)dsolve::SOLVER_RK_TYPE == RKType::RK45)
            ets->set_ets_coefficients(ts::ETSType::RK5);

        if (!rank) {
            std::cout << CYN << BLD << "Now initializing time stepper..." << NRM
                      << std::endl;
        }

        ets->init();

        if (!rank) {
            std::cout << GRN << BLD << "...Initialized!" << NRM << std::endl;
        }

#if defined __PROFILE_CTX__ && defined __PROFILE_ETS__
        std::ofstream outfile;
        char fname[256];
        sprintf(fname, "solverCtx_%d.txt", npes);
        if (!rank) {
            outfile.open(fname, std::ios_base::app);
            time_t now = time(0);
            // convert now to string form
            char* dt = ctime(&now);
            outfile << "======================================================="
                       "====="
                    << std::endl;
            outfile << "Current time : " << dt << " --- " << std::endl;
            outfile << "======================================================="
                       "====="
                    << std::endl;
        }

        ets->init_pt();
        solverCtx->reset_pt();
        ets->dump_pt(outfile);
        // solverCtx->dump_pt(outfile);
#endif

        // merging
        double t1 = MPI_Wtime();
        bool did_print_output_time = false;

        if (!rank) {
            std::cout << CYN << BLD << "Starting to evolve through time..."
                      << NRM << std::endl;
        }

        while (ets->curr_time() < dsolve::SOLVER_RK_TIME_END) {
            const DendroIntL step = ets->curr_step();
            const DendroScalar time = ets->curr_time();

            dsolve::SOLVER_CURRENT_RK_COORD_TIME = time;
            dsolve::SOLVER_CURRENT_RK_STEP = step;

            const bool isActive = ets->is_active();

            const unsigned int rank_global = ets->get_global_rank();

            if ((step % dsolve::SOLVER_REMESH_TEST_FREQ) == 0 && step != 0) {
                if (!rank_global)
                    std::cout << "[ETS] : Remesh time reached, checking to see "
                                 "if remesh should occur.  \n";

                bool isRemesh = solverCtx->is_remesh();
                if (isRemesh) {
                    if (!rank_global)
                        std::cout << "[ETS] : Remesh has been triggered.  \n";

                    solverCtx->remesh_and_gridtransfer(
                        dsolve::SOLVER_DENDRO_GRAIN_SZ,
                        dsolve::SOLVER_LOAD_IMB_TOL, dsolve::SOLVER_SPLIT_FIX);
                    dsolve::deallocate_deriv_workspace();
                    dsolve::allocate_deriv_workspace(solverCtx->get_mesh(), 1);
                    ets->sync_with_mesh();

                    ot::Mesh* pmesh = solverCtx->get_mesh();
                    unsigned int lmin, lmax;
                    pmesh->computeMinMaxLevel(lmin, lmax);
                    if (!rank_global)
                        printf("New min and max level = (%d, %d)\n", lmin,
                               lmax);
                    dsolve::SOLVER_RK45_TIME_STEP_SIZE =
                        dsolve::SOLVER_CFL_FACTOR *
                        ((dsolve::SOLVER_COMPD_MAX[0] -
                          dsolve::SOLVER_COMPD_MIN[0]) *
                         ((1u << (m_uiMaxDepth - lmax)) /
                          ((double)dsolve::SOLVER_ELE_ORDER)) /
                         ((double)(1u << (m_uiMaxDepth))));
                    ts::TSInfo ts_in = solverCtx->get_ts_info();
                    ts_in._m_uiTh = dsolve::SOLVER_RK45_TIME_STEP_SIZE;
                    solverCtx->set_ts_info(ts_in);
                } else {
                    if (!rank_global)
                        std::cout << "[ETS] : Remesh *not* triggered!.  \n";
                }
            }

            // NOTE: this is where the train,validate, etc. steps would go for
            // ML data
            did_print_output_time = false;

            if ((step % dsolve::SOLVER_TIME_STEP_OUTPUT_FREQ) == 0) {
                if (!rank_global)
                    std::cout << BLD << GRN << "==========\n"
                              << "[ETS - SOLVER] : SOLVER UPDATE\n"
                              << NRM << "\tCurrent Step: " << ets->curr_step()
                              << "\t\tCurrent time: " << ets->curr_time()
                              << "\tdt: " << ets->ts_size() << std::endl;

                solverCtx->terminal_output();
                did_print_output_time = true;
            }

            if ((step % dsolve::SOLVER_IO_OUTPUT_FREQ) == 0) {
                if (!rank_global) {
                    if (!did_print_output_time) {
                        std::cout << BLD << GRN << "==========\n"
                                  << "[ETS - SOLVER] : SOLVER UPDATE\n"
                                  << NRM
                                  << "\tCurrent Step: " << ets->curr_step()
                                  << "\t\tCurrent time: " << ets->curr_time()
                                  << "\tdt: " << ets->ts_size() << std::endl;
                    }

                    std::cout << BLD << BLU << "  --- NOW SAVING TO VTU" << NRM
                              << std::endl;
                    did_print_output_time = true;
                }

                solverCtx->write_vtu();
                if (!rank_global)
                    std::cout << BLD << GRN << "  --- FINISHED SAVING TO VTU"
                              << NRM << std::endl;
            }

            if ((step % dsolve::SOLVER_PROFILE_OUTPUT_FREQ) == 0) {
                if (!rank_global) {
                    if (!did_print_output_time) {
                        std::cout << BLD << GRN << "==========\n"
                                  << "[ETS - SOLVER] : SOLVER UPDATE\n"
                                  << NRM
                                  << "\tCurrent Step: " << ets->curr_step()
                                  << "\t\tCurrent time: " << ets->curr_time()
                                  << "\tdt: " << ets->ts_size() << std::endl;
                    }

                    std::cout << BLD << BLU << "  --- NOW WRITING PROFILE DATA"
                              << NRM << std::endl;
                    did_print_output_time = true;
                }

                dsolve::timer::profileInfoIntermediate(
                    dsolve::SOLVER_PROFILE_FILE_PREFIX.c_str(),
                    solverCtx->get_mesh(), step);

                if (!rank_global)
                    std::cout << BLD << GRN
                              << "  --- FINISHED WRITING PROFILE DATA" << NRM
                              << std::endl;
            }

            if ((step % dsolve::SOLVER_CHECKPT_FREQ) == 0)
                solverCtx->write_checkpt();

            dsolve::timer::t_rkStep.start();
            ets->evolve();
            dsolve::timer::t_rkStep.stop();
            solverCtx->resetForNextStep();
        }

#if defined __PROFILE_CTX__ && defined __PROFILE_ETS__
        ets->dump_pt(outfile);
        // solverCtx->dump_pt(outfile);
#endif

        double t2 = MPI_Wtime() - t1;
        double t2_g;
        par::Mpi_Allreduce(&t2, &t2_g, 1, MPI_MAX, ets->get_global_comm());
        if (!(ets->get_global_rank()))
            std::cout << " ETS time (max) : " << t2_g << std::endl;

        // cleanup
        ot::Mesh* tmp_mesh = solverCtx->get_mesh();
        delete solverCtx;
        delete tmp_mesh;
        delete ets;

    } else {
        // ========================================
        // OLD method of solver
        ode::solver::RK_SOLVER rk_solver(
            mesh, dsolve::SOLVER_RK_TIME_BEGIN, dsolve::SOLVER_RK_TIME_END,
            dsolve::SOLVER_RK45_TIME_STEP_SIZE, (RKType)dsolve::SOLVER_RK_TYPE);

        if (dsolve::SOLVER_RESTORE_SOLVER == 1) {
            rk_solver.restoreCheckPoint(
                dsolve::SOLVER_CHKPT_FILE_PREFIX.c_str(), comm);
        }

        /**-
         * STEP 5
         *
         * Start the solver!
         * This runs the main loop of the program and begins solving everything
         */
        if (!rank) {
            std::cout << GRN << "Now starting solver!" << NRM << std::endl;
        }

        dsolve::timer::t_rkSolve.start();
        rk_solver.rkSolve();
        dsolve::timer::t_rkSolve.stop();

        /**
         * STEP 6
         *
         * Solver finished, finalize everything and prepare for exiting the
         * program
         */
        dsolve::timer::total_runtime.stop();

        double t2 = dsolve::timer::t_rkSolve.seconds;
        double t2_g;
        par::Mpi_Allreduce(&t2, &t2_g, 1, MPI_MAX, comm);
        if (!rank) {
            std::cout << "  SOLVER TIME (max): " << t2_g << std::endl;
        }

        rk_solver.freeMesh();
    }

    MPI_Finalize();

    if (!rank) {
        std::cout << GRN << "Solver finished!" << NRM << std::endl;
    }

    return 0;
}
