//
// Created by milinda on 7/26/17.
/**
 *@author Milinda Fernando
 *School of Computing, University of Utah
 *@brief Contains utility functions for SOLVER simulation.
 */
//

// TODO: multiple things to do here, but the big idea is that the puncture needs
// to be defined *somehow* for SOLVER I worry that if we change var names then
// we'll have to edit this stuff by hand

// TODO: read param file needs to be modified whenever there are new params as
// well

#include "grUtils.h"

#include "compact_derivs.h"

namespace dsolve {

// NOTE: the read param file and dump param file are now included in
// parameters.cpp

void initDataFuncToPhysCoords(const double xx1, const double yy1,
                              const double zz1, double *var) {
    // convert "grid" values to physical values
    const double xx = GRIDX_TO_X(xx1);
    const double yy = GRIDY_TO_Y(yy1);
    const double zz = GRIDZ_TO_Z(zz1);

    switch (dsolve::SOLVER_ID_TYPE) {
        case 0:

            initDataEM4(xx, yy, zz, var);
            break;

        default:
            std::cout << "Unknown ID type: " << dsolve::SOLVER_ID_TYPE
                      << std::endl;
            exit(0);
            break;
    }
}

void initDataEM4(const double x, const double y, const double z, double *var) {
    // NOTE: remember that x, y, and z are already CONVERTED from grid!
    const double amp1 = dsolve::EM4_ID_AMP1;
    const double lambda1 = dsolve::EM4_ID_LAMBDA1;

    double B0,B1,B2, E0,E1,E2, Phi,Psi ; 
    double rho_e, J0,J1,J2;
    double r,Ephi_over_r,tmp_Ephiup ; 

    r = sqrt( x*x + y*y + z*z ) ; 
    tmp_Ephiup = - 8.0*amp1*lambda1*lambda1*exp(-lambda1*r*r) ; 
    E0 = - y * tmp_Ephiup ; 
    E1 =   x * tmp_Ephiup ; 
    E2 = 0.0 ; 

    B0 = 0.0 ;  
    B1 = 0.0 ;  
    B2 = 0.0 ;  
    
    Phi = 0.0 ;  
    Psi = 0.0 ;  
    
    J0 = 0.0 ;  
    J1 = 0.0 ;  
    J2 = 0.0 ;  
    
    rho_e = 0.0 ;  
    
    var[VAR::U_E0] = E0;
    var[VAR::U_E1] = E1;
    var[VAR::U_E2] = E2;
    var[VAR::U_B0] = B0;
    var[VAR::U_B1] = B1;
    var[VAR::U_B2] = B2;
    var[VAR::U_PHI] = Phi;
    var[VAR::U_PSI] = Psi;

}

double CalTolHelper(const double t, const double r, const double rad[],
                    const double eps[], const double toffset) {
    const double R0 = rad[0];
    const double R1 = rad[1];
    const double RGW = rad[2];
    const double tol = eps[0];
    const double tolGW = eps[1];
    const double tolMax = eps[2];
    const double WRR = std::min(tolGW, tolMax);
    if (r < R0) {
        return tol;
    }
    if (t > (R1 + toffset)) {
        const double RR = std::min(t - toffset, RGW + 10.0);
        const double WTolExpFac = (RR - R0) / log10(WRR / tol);
        return std::min(tolMax, tol * pow(10.0, ((r - R0) / WTolExpFac)));
    } else {
        const double WTolExpFac = (R1 - R0) / log10(WRR / tol);
        return std::min(tolMax, tol * pow(10.0, ((r - R0) / WTolExpFac)));
    }
}

void analyticalSolEM4_BLOCK(double **uZipAnalyticVars,
                            double **uZipAnalyticDiffVars,
                            const double **uZipVars, const double time,
                            const unsigned int &offset, const double *pmin,
                            const double *pmax, const unsigned int *sz,
                            const unsigned int &bflag) {
    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];
    const unsigned int n = nx * ny * nz;
    unsigned int BLK_SZ = n;

    const double hx = (pmax[0] - pmin[0]) / (nx - 1);
    const double hy = (pmax[1] - pmin[1]) / (ny - 1);
    const double hz = (pmax[2] - pmin[2]) / (nz - 1);

    const unsigned int PW = dsolve::SOLVER_PADDING_WIDTH;

    // The true variables
    const double *E0 = &uZipVars[VAR::U_E0][offset];
    const double *E1 = &uZipVars[VAR::U_E1][offset];
    const double *E2 = &uZipVars[VAR::U_E2][offset];
    const double *B0 = &uZipVars[VAR::U_B0][offset];
    const double *B1 = &uZipVars[VAR::U_B1][offset];
    const double *B2 = &uZipVars[VAR::U_B2][offset];
    const double *Phi = &uZipVars[VAR::U_PHI][offset];
    const double *Psi = &uZipVars[VAR::U_PSI][offset];


    // the analytic variables
    double *E0_a = &uZipAnalyticVars[VAR::U_E0][offset];
    double *E1_a = &uZipAnalyticVars[VAR::U_E1][offset];
    double *E2_a = &uZipAnalyticVars[VAR::U_E2][offset];
    double *B0_a = &uZipAnalyticVars[VAR::U_B0][offset];
    double *B1_a = &uZipAnalyticVars[VAR::U_B1][offset];
    double *B2_a = &uZipAnalyticVars[VAR::U_B2][offset];
    double *Phi_a = &uZipAnalyticVars[VAR::U_PHI][offset];
    double *Psi_a = &uZipAnalyticVars[VAR::U_PSI][offset];

    // the analytic difference variables
    double *E0_adiff = &uZipAnalyticDiffVars[VAR::U_E0][offset];
    double *E1_adiff = &uZipAnalyticDiffVars[VAR::U_E1][offset];
    double *E2_adiff = &uZipAnalyticDiffVars[VAR::U_E2][offset];
    double *B0_adiff = &uZipAnalyticDiffVars[VAR::U_B0][offset];
    double *B1_adiff = &uZipAnalyticDiffVars[VAR::U_B1][offset];
    double *B2_adiff = &uZipAnalyticDiffVars[VAR::U_B2][offset];
    double *Phi_adiff = &uZipAnalyticDiffVars[VAR::U_PHI][offset];
    double *Psi_adiff = &uZipAnalyticDiffVars[VAR::U_PSI][offset];

    // then loop through
    double varinput[dsolve::SOLVER_NUM_VARS];

    for (unsigned int k = PW; k < nz - PW; k++) {
        for (unsigned int j = PW; j < ny - PW; j++) {
#ifdef SOLVER_ENABLE_AVX
#ifdef __INTEL_COMPILER
#pragma vector vectorlength(__RHS_AVX_SIMD_LEN__) vecremainder
#pragma ivdep
#endif
#endif
            for (unsigned int i = PW; i < nx - PW; i++) {
                const double x = pmin[0] + i * hx;
                const double y = pmin[1] + j * hy;
                const double z = pmin[2] + k * hz;
                const unsigned int pp = i + nx * (j + ny * k);

                analyticalSolEM4(x, y, z, time, varinput, false);

                // extract out the values
                E0_a[pp] = varinput[VAR::U_E0];
                E1_a[pp] = varinput[VAR::U_E1];
                E2_a[pp] = varinput[VAR::U_E2];
                B0_a[pp] = varinput[VAR::U_B0];
                B1_a[pp] = varinput[VAR::U_B1];
                B2_a[pp] = varinput[VAR::U_B2];
                Phi_a[pp] = varinput[VAR::U_PHI];
                Psi_a[pp] = varinput[VAR::U_PSI];

                // calculate the difference
                E0_adiff[pp] = E0_a[pp] - E0[pp];
                E1_adiff[pp] = E1_a[pp] - E1[pp];
                E2_adiff[pp] = E2_a[pp] - E2[pp];

                B0_adiff[pp] = B0_a[pp] - B0[pp];
                B1_adiff[pp] = B1_a[pp] - B1[pp];
                B2_adiff[pp] = B2_a[pp] - B2[pp];
                Phi_adiff[pp] = Phi_a[pp] - Phi[pp];
                Psi_adiff[pp] = Psi_a[pp] - Psi[pp];
            }
        }
    }
    // finished with the analytic solve
}

// added analytic calculation for the dipole pulse -AJC
void analyticalSolEM4(const double xx, const double yy, const double zz,
                      const double t, double *var, bool varsAreGrid) {
    double x, y, z;
    if (varsAreGrid) {
        x = GRIDX_TO_X(xx);
        y = GRIDY_TO_Y(yy);
        z = GRIDZ_TO_Z(zz);
    } else {
        x = xx;
        y = yy;
        z = zz;
    }

    const double amp1 = dsolve::EM4_ID_AMP1;
        const double lambda1 = dsolve::EM4_ID_LAMBDA1;

        double B0,B1,B2, E0,E1,E2, Phi,Psi ;
        double rho_e, J0,J1,J2;
        double r,tmp_Ephiup, tmp_Br_up, tmp_Btheta_up ;

        r = sqrt( x*x + y*y + z*z ) ;
        if ( r < 1.e-8 ) { r=1.e-8 ; }

        tmp_Br_up = - 2.0 * lambda1
                          * (   (t-r)*exp( - lambda1*(t-r)*(t-r) )
                              + (t+r)*exp( - lambda1*(t+r)*(t+r) ) ) / (r*r)
                    + (   exp( - lambda1*(t-r)*(t-r) )
                        - exp( - lambda1*(t+r)*(t+r) ) ) / (r*r*r)
                   ;
        tmp_Br_up *= 2.0 * amp1 ;  

        tmp_Btheta_up = - 2.0*lambda1
                             * (   exp( - lambda1*(t-r)*(t-r) )
                                 - exp( - lambda1*(t+r)*(t+r) ) ) / r
                        + 4.0*lambda1*lambda1
                             * (   (t-r)*(t-r)*exp( - lambda1*(t-r)*(t-r) )
                                 - (t+r)*(t+r)*exp( - lambda1*(t+r)*(t+r) ) ) 
                             / r
                        - 2.0*lambda1
                             * (   (t-r)*exp( - lambda1*(t-r)*(t-r) )
                                 + (t+r)*exp( - lambda1*(t+r)*(t+r) ) ) / (r*r)
                       + (   exp( - lambda1*(t-r)*(t-r) )
                           - exp( - lambda1*(t+r)*(t+r) ) ) / (r*r*r)
                   ;
        tmp_Btheta_up *= amp1 ;  

        tmp_Ephiup =   2.0*amp1*lambda1
                          * (   (t-r)*exp( - lambda1*(t-r)*(t-r) )
                              - (t+r)*exp( - lambda1*(t+r)*(t+r) ) ) / (r*r)
                     + 2.0*amp1*lambda1
                          * (   exp( - lambda1*(t-r)*(t-r) )
                              + exp( - lambda1*(t+r)*(t+r) ) ) / r
                     - 4.0*amp1*lambda1*lambda1
                          * (   (t-r)*(t-r)*exp( - lambda1*(t-r)*(t-r) )
                              + (t+r)*(t+r)*exp( - lambda1*(t+r)*(t+r) ) ) / r
                   ;

        E0 = - y * tmp_Ephiup / r ;
        E1 =   x * tmp_Ephiup / r ;
        E2 = 0.0 ;

        B0 = x*z * ( tmp_Br_up + tmp_Btheta_up ) / (r*r) ;
        B1 = y*z * ( tmp_Br_up + tmp_Btheta_up ) / (r*r) ;
        B2 = ( z*z * tmp_Br_up - (x*x+y*y) * tmp_Btheta_up ) / (r*r) ;

        Phi = 0.0 ;
        Psi = 0.0 ;

        J0 = 0.0 ;
        J1 = 0.0 ;
        J2 = 0.0 ;

        var[VAR::U_E0] = E0 ;
        var[VAR::U_E1] = E1 ;
        var[VAR::U_E2] = E2 ;
        var[VAR::U_B0] = B0 ;
        var[VAR::U_B1] = B1 ;
        var[VAR::U_B2] = B2 ;
        var[VAR::U_PHI] = Phi ;
        var[VAR::U_PSI] = Psi ;
}

void blockAdaptiveOctree(std::vector<ot::TreeNode> &tmpNodes,
                         const Point &pt_min, const Point &pt_max,
                         const unsigned int regLev, const unsigned int maxDepth,
                         MPI_Comm comm) {
    int rank, npes;
    MPI_Comm_size(comm, &npes);
    MPI_Comm_rank(comm, &rank);

    double pt_g_min[3];
    double pt_g_max[3];

    pt_g_min[0] = X_TO_GRIDX(pt_min.x());
    pt_g_min[1] = Y_TO_GRIDY(pt_min.y());
    pt_g_min[2] = Z_TO_GRIDZ(pt_min.z());

    pt_g_max[0] = X_TO_GRIDX(pt_max.x());
    pt_g_max[1] = Y_TO_GRIDY(pt_max.y());
    pt_g_max[2] = Z_TO_GRIDZ(pt_max.z());

    assert(pt_g_min[0] >= 0 && pt_g_min[0] <= (1u << maxDepth));
    assert(pt_g_min[1] >= 0 && pt_g_min[1] <= (1u << maxDepth));
    assert(pt_g_min[2] >= 0 && pt_g_min[2] <= (1u << maxDepth));

    assert(pt_g_max[0] >= 0 && pt_g_max[0] <= (1u << maxDepth));
    assert(pt_g_max[1] >= 0 && pt_g_max[1] <= (1u << maxDepth));
    assert(pt_g_max[2] >= 0 && pt_g_max[2] <= (1u << maxDepth));

    unsigned int xRange_b, xRange_e;
    unsigned int yRange_b = pt_g_min[1], yRange_e = pt_g_max[1];
    unsigned int zRange_b = pt_g_min[2], zRange_e = pt_g_max[2];

    xRange_b =
        pt_g_min[0];  //(rank*(pt_g_max[0]-pt_g_min[0]))/npes + pt_g_min[0];
    xRange_e = pt_g_max[1];  //((rank+1)*(pt_g_max[0]-pt_g_min[0]))/npes +
                             // pt_g_min[0];

    unsigned int stepSz = 1u << (maxDepth - regLev);

    /* std::cout<<" x min: "<<xRange_b<<" x_max: "<<xRange_e<<std::endl;
    std::cout<<" y min: "<<yRange_b<<" y_max: "<<yRange_e<<std::endl;
    std::cout<<" z min: "<<zRange_b<<" z_max: "<<zRange_e<<std::endl;*/

    // std::cout << "Now adjusting tmpNodes" << std::endl;

    for (unsigned int x = xRange_b; x < xRange_e; x += stepSz)
        for (unsigned int y = yRange_b; y < yRange_e; y += stepSz)
            for (unsigned int z = zRange_b; z < zRange_e; z += stepSz) {
                if (x >= (1u << maxDepth)) x = x - 1;
                if (y >= (1u << maxDepth)) y = y - 1;
                if (z >= (1u << maxDepth)) z = z - 1;

                tmpNodes.push_back(
                    ot::TreeNode(x, y, z, regLev, m_uiDim, maxDepth));
            }

    return;
}

double computeWTol(double x, double y, double z, double tolMin) {
    double origin[3];
    origin[0] = (double)((1u << dsolve::SOLVER_MAXDEPTH) - 1);
    origin[1] = (double)((1u << dsolve::SOLVER_MAXDEPTH) - 1);
    origin[2] = (double)((1u << dsolve::SOLVER_MAXDEPTH) - 1);

    double r =
        sqrt(GRIDX_TO_X(x) * GRIDX_TO_X(x) + GRIDY_TO_Y(y) * GRIDY_TO_Y(y) +
             GRIDZ_TO_Z(z) * GRIDZ_TO_Z(z));

    const double tolMax = dsolve::SOLVER_WAVELET_TOL_MAX;
    const double R0 = dsolve::SOLVER_WAVELET_TOL_FUNCTION_R0;
    const double R1 = dsolve::SOLVER_WAVELET_TOL_FUNCTION_R1;

    if (dsolve::SOLVER_USE_WAVELET_TOL_FUNCTION == 1) {
        return std::min(
            tolMax, std::max(tolMin, (tolMax - tolMin) / (R1 - R0) * (r - R0) +
                                         tolMin));
    } else {
        return tolMin;
    }
}

double computeWTolDCoords(double x, double y, double z, double *hx) {
    const unsigned int eleOrder = dsolve::SOLVER_ELE_ORDER;

    if (dsolve::SOLVER_USE_WAVELET_TOL_FUNCTION == 0) {
        return dsolve::SOLVER_WAVELET_TOL;

    } else {
        return dsolve::SOLVER_WAVELET_TOL;
    }
}

void writeBLockToBinary(const double **unzipVarsRHS, unsigned int offset,
                        const double *pmin, const double *pmax, double *bxMin,
                        double *bxMax, const unsigned int *sz,
                        unsigned int blkSz, double dxFactor,
                        const char *fprefix) {
    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    const unsigned int ib = 3;
    const unsigned int jb = 3;
    const unsigned int kb = 3;

    const unsigned int ie = nx - 3;
    const unsigned int je = ny - 3;
    const unsigned int ke = nz - 3;

    const unsigned int blkInlSz = (nx - 3) * (ny - 3) * (nz - 3);

    double hx = (pmax[0] - pmin[0]) / (nx - 1);
    double hy = (pmax[1] - pmin[1]) / (ny - 1);
    double hz = (pmax[2] - pmin[2]) / (nz - 1);

    const double dx =
        (dsolve::SOLVER_COMPD_MAX[0] - dsolve::SOLVER_COMPD_MIN[0]) *
        (1.0 / (double)(1u << dsolve::SOLVER_MAXDEPTH));
    unsigned int level =
        dsolve::SOLVER_MAXDEPTH - ((unsigned int)(hx / dx) - 1);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, npes;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);
    // std::cout<<"ranl: "<<rank<<"npes: "<<npes<<std::endl;

    // std::cout<<"nx: "<<nx<<" level: "<<level<<" hx: "<<hx<<" dx:
    // "<<dx<<std::endl;

    if ((hx > (dxFactor * dx)) ||
        (pmin[0] < bxMin[0] || pmin[1] < bxMin[1] || pmin[2] < bxMin[2]) ||
        (pmax[0] > bxMax[0] || pmax[1] > bxMax[1] || pmax[2] > bxMax[2]))
        return;

    double *blkInternal = new double[blkInlSz];
    for (unsigned int var = 0; var < dsolve::SOLVER_NUM_VARS; var++) {
        char fName[256];
        sprintf(fName, "%s_%s_n_%d_r_%d_p_%d.bin", fprefix,
                dsolve::SOLVER_VAR_NAMES[var], nx, rank, npes);
        FILE *outfile = fopen(fName, "w");
        if (outfile == NULL) {
            std::cout << fName << " file open failed " << std::endl;
        }

        for (unsigned int k = kb; k < ke; k++)
            for (unsigned int j = jb; j < je; j++)
                for (unsigned int i = ib; i < ie; i++)
                    blkInternal[k * (ny - 3) * (nx - 3) + j * (nx - 3) + i] =
                        unzipVarsRHS[var]
                                    [offset + k * (ny * nx) + j * (ny) + i];

        fwrite(blkInternal, sizeof(double), blkInlSz,
               outfile);  // write out the number of elements.
        fclose(outfile);
    }

    delete[] blkInternal;
}

unsigned int getOctantWeight(const ot::TreeNode *pNode) {
    return (1u << (3 * pNode->getLevel())) * 1;
}

void allocate_deriv_workspace(const ot::Mesh *pMesh, unsigned int s_fac) {
    // start with deallocation of the workspace, needed when remeshing
    deallocate_deriv_workspace();

    if (!pMesh->isActive()) return;

    // then get the largest block size from the mesh
    const std::vector<ot::Block> &blkList = pMesh->getLocalBlockList();
#if EM4_ENABLE_COMPACT_DERIVS
    std::vector<unsigned int> unique_block_sizes;
    unique_block_sizes.reserve(blkList.size());
#endif
    unsigned int max_blk_sz = 0;
    for (unsigned int i = 0; i < blkList.size(); i++) {
        unsigned int blk_sz = blkList[i].getAllocationSzX() *
                              blkList[i].getAllocationSzY() *
                              blkList[i].getAllocationSzZ();
        if (blk_sz > max_blk_sz) max_blk_sz = blk_sz;

#if EM4_ENABLE_COMPACT_DERIVS
        unique_block_sizes.push_back(blkList[i].getAllocationSzX());
        unique_block_sizes.push_back(blkList[i].getAllocationSzY());
        unique_block_sizes.push_back(blkList[i].getAllocationSzZ());
#endif
    }

    // make sure the derivatives are deallocated? seems unnecessary since
    // it's done earlier?
    deallocate_deriv_workspace();

#ifdef EM4_ENABLE_COMPACT_DERIVS
    // use this opportunity to tell the CFD object just how big things are here!
    dendro_cfd::cfd.initialize_cfd_3dblock_workspace(max_blk_sz);
#endif

    // allocate the new memory
    dsolve::SOLVER_DERIV_WORKSPACE =
        new double[s_fac * max_blk_sz * dsolve::SOLVER_NUM_DERIVATIVES];

#ifdef EM4_ENABLE_COMPACT_DERIVS
    // OLD CODE:
    // then we want to make sure we've got the unique sizes set up
    std::sort(unique_block_sizes.begin(), unique_block_sizes.end());
    std::vector<unsigned int>::iterator it;
    it = std::unique(unique_block_sizes.begin(), unique_block_sizes.end());
    unique_block_sizes.resize(distance(unique_block_sizes.begin(), it));

    // iterate through them
    for (unsigned int i = 0; i < unique_block_sizes.size(); i++) {
        dendro_cfd::cfd.add_cfd_matrix_to_storage(unique_block_sizes[i]);
        // std::cout << "adding size: " << unique_block_sizes[i] << " to cfd"
        //           << std::endl;
    }

#ifdef DENDRO_USE_NEW_DERIVS
    // make sure the maximum block size is properly set
    SOLVER_DERIVS->set_maximum_block_size(max_blk_sz);
#endif

#endif
}

void deallocate_deriv_workspace() {
    // if the pointer isn't already null, delete the allocated memory
    if (dsolve::SOLVER_DERIV_WORKSPACE != nullptr) {
        delete[] dsolve::SOLVER_DERIV_WORKSPACE;
        dsolve::SOLVER_DERIV_WORKSPACE = nullptr;
    }

#ifdef EM4_ENABLE_COMPACT_DERIVS
    // use this opportunity to tell the CFD object just how big things are here!
    dendro_cfd::cfd.delete_cfd_3dblock_workspace();
#endif
}

ot::Mesh *weakScalingReMesh(ot::Mesh *pMesh, unsigned int target_npes) {
    int rank, npes;

    MPI_Comm comm = pMesh->getMPIGlobalCommunicator();
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);

    if (target_npes > npes) {
        if (!rank)
            RAISE_ERROR("target npes "
                        << target_npes
                        << " is larger than global npes:" << npes);

        MPI_Abort(comm, 0);
    }

    const double R_RES_FAC = 10;

    const DendroIntL ELE_SZ_REQ_GL =
        dsolve::SOLVER_DENDRO_GRAIN_SZ * target_npes;
    const unsigned int MAX_ITER = 30;

    unsigned int iter_count = 0;
    bool is_converged = true;

    // Sequential split of each level, such that elemental grain size is
    // satiesfied.
    ot::Mesh *current_mesh = NULL;

    do {
        if (current_mesh == NULL) current_mesh = pMesh;

        unsigned int LMIN, LMAX;
        current_mesh->computeMinMaxLevel(LMIN, LMAX);

        DendroIntL localSz = current_mesh->getNumLocalMeshElements();
        DendroIntL globalSz = 0;

        par::Mpi_Allreduce(&localSz, &globalSz, 1, MPI_SUM, comm);

        DendroIntL req_l_splits =
            std::max(1ll, (ELE_SZ_REQ_GL - globalSz) / (14 * npes));

        std::vector<unsigned int> ref_flags;
        if (current_mesh->isActive()) {
            ref_flags.resize(current_mesh->getNumLocalMeshElements(),
                             OCT_NO_CHANGE);
            const unsigned int active_rank = current_mesh->getMPIRank();
            const unsigned int active_npes = current_mesh->getMPICommSize();
            const MPI_Comm active_comm = current_mesh->getMPICommunicator();
            // const DendroIntL lsplit_b =
            // (active_rank)*req_l_splits/active_npes; const DendroIntL lsplit_e
            // = (active_rank+1)*req_l_splits/active_npes; const DendroIntL
            // num_split_rank = lsplit_e-lsplit_b;

            const ot::TreeNode *pNodes = current_mesh->getAllElements().data();

            // DendroIntL lcount =0;
            // for (unsigned int ele = current_mesh->getElementLocalBegin(); ele
            // < current_mesh->getElementLocalEnd(); ele++)
            // {
            //     if(pNodes[ele].getLevel() == LMAX-1)
            //         lcount++;
            // }

            // std::vector<DendroIntL> num_lev_counts;
            // std::vector<DendroIntL> num_lev_offsets;

            // num_lev_counts.resize(active_npes,0);
            // num_lev_offsets.resize(active_npes,0);

            // par::Mpi_Allgather(&lcount,num_lev_counts.data(),1,active_comm);
            // omp_par::scan(num_lev_counts.data(),num_lev_offsets.data(),active_npes);

            // auto it_upper =
            // std::upper_bound(num_lev_offsets.begin(),num_lev_offsets.end(),req_l_splits);
            // unsigned int valid_npes=active_npes;

            // if(it_upper != num_lev_offsets.end())
            //     valid_npes = std::distance(num_lev_offsets.begin(),it_upper)
            //     +1;

            // if (active_rank < valid_npes)
            // {
            //     unsigned int ele_offset =
            //     current_mesh->getElementLocalBegin(); lcount =0; for
            //     (unsigned int ele = current_mesh->getElementLocalBegin(); ele
            //     < current_mesh->getElementLocalEnd(); ele++)
            //     {
            //         if( lcount < req_l_splits  && pNodes[ele].getLevel() ==
            //         LMAX-1)
            //         {
            //             ref_flags[ele-ele_offset] = OCT_SPLIT;
            //             lcount++;
            //         }

            //     }

            // }
            unsigned int ele_offset = current_mesh->getElementLocalBegin();
            DendroIntL lcount = 0;
            for (unsigned int ele = current_mesh->getElementLocalBegin();
                 ele < current_mesh->getElementLocalEnd(); ele++) {
                if (lcount < req_l_splits &&
                    pNodes[ele].getLevel() == LMAX - 1) {
                    ref_flags[ele - ele_offset] = OCT_SPLIT;
                    lcount++;
                }
            }
        }

        bool is_refine = current_mesh->setMeshRefinementFlags(ref_flags);
        bool is_refine_g = false;
        MPI_Allreduce(&is_refine, &is_refine_g, 1, MPI_CXX_BOOL, MPI_LOR, comm);
        if (is_refine_g) {
            ot::Mesh *new_mesh =
                current_mesh->ReMesh(dsolve::SOLVER_DENDRO_GRAIN_SZ);
            if (current_mesh == pMesh)
                current_mesh = new_mesh;
            else {
                std::swap(current_mesh, new_mesh);
                delete new_mesh;
            }

            localSz = current_mesh->getNumLocalMeshElements();
            par::Mpi_Allreduce(&localSz, &globalSz, 1, MPI_SUM, comm);

            if (!rank)
                std::cout << "weak scaling remesh iter : " << iter_count
                          << " num global elements : " << globalSz << std::endl;

            is_converged =
                (globalSz >=
                 ELE_SZ_REQ_GL);  //|| (fabs(globalSz/(double)npes -
                                  // bssn::BSSN_DENDRO_GRAIN_SZ)/(double)bssn::BSSN_DENDRO_GRAIN_SZ
                                  //< 0.1);//(globalSz >= ELE_SZ_REQ_GL);
            iter_count++;

        } else {
            // no remesh triggered hence, is_converged true;
            is_converged = true;
        }

    } while (!is_converged && iter_count < MAX_ITER);

    return current_mesh;
}

}  // end of namespace dsolve

namespace dsolve {

namespace timer {
void initFlops() {
    total_runtime.start();
    t_f2o.start();
    t_cons.start();
    t_bal.start();
    t_mesh.start();
    t_rkSolve.start();
    t_ghostEx_sync.start();
    t_unzip_sync.start();

    for (unsigned int i = 0; i < NUM_FACES; i++)
        dendro::timer::t_unzip_sync_face[i].start();

    dendro::timer::t_unzip_async_internal.start();
    dendro::timer::t_unzip_sync_edge.start();
    dendro::timer::t_unzip_sync_vtex.start();
    dendro::timer::t_unzip_p2c.start();
    dendro::timer::t_unzip_sync_nodalval.start();
    dendro::timer::t_unzip_sync_cpy.start();
    dendro::timer::t_unzip_sync_f_c1.start();
    dendro::timer::t_unzip_sync_f_c2.start();
    dendro::timer::t_unzip_sync_f_c3.start();

    t_unzip_async.start();
    dendro::timer::t_unzip_async_comm.start();

    dendro::timer::t_unzip_async_internal.start();
    dendro::timer::t_unzip_async_external.start();
    dendro::timer::t_unzip_async_comm.start();
    t_deriv.start();
    t_rhs.start();

    // t_rhs_a.start();
    // t_rhs_b.start();
    // t_rhs_gt.start();
    // t_rhs_chi.start();
    // t_rhs_At.start();
    // t_rhs_K.start();
    // t_rhs_Gt.start();
    // t_rhs_B.start();
    //
    t_bdyc.start();

    t_zip.start();
    t_rkStep.start();
    t_isReMesh.start();
    t_gridTransfer.start();
    t_ioVtu.start();
    t_ioCheckPoint.start();
}

void resetSnapshot() {
    total_runtime.snapreset();
    t_f2o.snapreset();
    t_cons.snapreset();
    t_bal.snapreset();
    t_mesh.snapreset();
    t_rkSolve.snapreset();
    t_ghostEx_sync.snapreset();
    t_unzip_sync.snapreset();

    for (unsigned int i = 0; i < NUM_FACES; i++)
        dendro::timer::t_unzip_sync_face[i].snapreset();

    dendro::timer::t_unzip_sync_internal.snapreset();
    dendro::timer::t_unzip_sync_edge.snapreset();
    dendro::timer::t_unzip_sync_vtex.snapreset();
    dendro::timer::t_unzip_p2c.snapreset();
    dendro::timer::t_unzip_sync_nodalval.snapreset();
    dendro::timer::t_unzip_sync_cpy.snapreset();

    dendro::timer::t_unzip_sync_f_c1.snapreset();
    dendro::timer::t_unzip_sync_f_c2.snapreset();
    dendro::timer::t_unzip_sync_f_c3.snapreset();

    t_unzip_async.snapreset();
    dendro::timer::t_unzip_async_internal.snapreset();
    dendro::timer::t_unzip_async_external.snapreset();
    dendro::timer::t_unzip_async_comm.snapreset();

    t_deriv.snapreset();
    t_rhs.snapreset();

    // t_rhs_a.snapreset();
    // t_rhs_b.snapreset();
    // t_rhs_gt.snapreset();
    // t_rhs_chi.snapreset();
    // t_rhs_At.snapreset();
    // t_rhs_K.snapreset();
    // t_rhs_Gt.snapreset();
    // t_rhs_B.snapreset();
    //
    t_bdyc.snapreset();

    t_zip.snapreset();
    t_rkStep.snapreset();
    t_isReMesh.snapreset();
    t_gridTransfer.snapreset();
    t_ioVtu.snapreset();
    t_ioCheckPoint.snapreset();
}

void profileInfo(const char *filePrefix, const ot::Mesh *pMesh) {
    int activeRank, activeNpes, globalRank, globalNpes;

    MPI_Comm commActive;
    MPI_Comm commGlobal;

    if (pMesh->isActive()) {
        commActive = pMesh->getMPICommunicator();
        activeRank = pMesh->getMPIRank();
        activeNpes = pMesh->getMPICommSize();
    } else {
        return;
    }

    globalRank = pMesh->getMPIRankGlobal();
    globalNpes = pMesh->getMPICommSizeGlobal();
    commGlobal = pMesh->getMPIGlobalCommunicator();

    double t_stat;
    double t_stat_g[3];

    const char separator = ' ';
    const int nameWidth = 30;
    const int numWidth = 10;

    char fName[256];
    std::ofstream outfile;

    DendroIntL localSz, globalSz;

    if (!activeRank) {
        sprintf(fName, "%s_final.prof", filePrefix);
        outfile.open(fName);
        if (outfile.fail()) {
            std::cout << fName << " file open failed " << std::endl;
            return;
        }

        outfile << "active npes : " << activeNpes << std::endl;
        outfile << "global npes : " << globalNpes << std::endl;
        outfile << "partition tol : " << dsolve::SOLVER_LOAD_IMB_TOL
                << std::endl;
        outfile << "wavelet tol : " << dsolve::SOLVER_WAVELET_TOL << std::endl;
        outfile << "maxdepth : " << dsolve::SOLVER_MAXDEPTH << std::endl;
    }

    MPI_Comm comm = commActive;
    unsigned int rank = activeRank;

    localSz = pMesh->getNumLocalMeshElements();
    par::Mpi_Reduce(&localSz, &globalSz, 1, MPI_SUM, 0, comm);
    if (!rank) outfile << "Elements : " << globalSz << std::endl;

    localSz = pMesh->getNumLocalMeshNodes();
    par::Mpi_Reduce(&localSz, &globalSz, 1, MPI_SUM, 0, comm);
    if (!rank) outfile << "DOG(zip) : " << globalSz << std::endl;

    localSz = pMesh->getDegOfFreedomUnZip();
    par::Mpi_Reduce(&localSz, &globalSz, 1, MPI_SUM, 0, comm);
    if (!rank) outfile << "DOG(unzip) : " << globalSz << std::endl;

    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "step";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "min(s)";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "mean(s)";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "max(s)" << std::endl;

    t_stat = total_runtime.seconds;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "+runtime(s)";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    t_stat = t_f2o.seconds;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << " ++f2o";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    t_stat = t_cons.seconds;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << " ++construction";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    t_stat = t_rkSolve.seconds;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << " ++rkSolve";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    t_stat = t_bal.seconds;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "  --2:1 balance";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    t_stat = t_mesh.seconds;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "  --mesh";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    t_stat = t_rkStep.seconds;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "  --rkstep";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    t_stat = t_ghostEx_sync.seconds;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "  --ghostExchge.";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    t_stat = t_unzip_sync.seconds;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "  --unzip_sync";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    t_stat = t_unzip_async.seconds;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "  ++unzip_async";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
    t_stat = dendro::timer::t_unzip_async_internal.seconds;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "  --unzip_internal";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    t_stat = dendro::timer::t_unzip_async_external.seconds;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "  --unzip_external";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    t_stat = dendro::timer::t_unzip_async_comm.seconds;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "  --unzip_comm (comm) ";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;
#endif

    t_stat = t_deriv.seconds;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "  --deriv ";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    t_stat = t_rhs.seconds;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "  --compute_rhs ";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    t_stat = t_bdyc.seconds;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "  --boundary con ";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    t_stat = t_zip.seconds;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "  --zip";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    t_stat = t_ioVtu.seconds;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "  --vtu";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    t_stat = t_ioCheckPoint.seconds;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "  --checkpoint";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    if (!rank) outfile.close();
}

void profileInfoIntermediate(const char *filePrefix, const ot::Mesh *pMesh,
                             const unsigned int currentStep) {
    int activeRank, activeNpes, globalRank, globalNpes;

    MPI_Comm commActive;
    MPI_Comm commGlobal;

    if (pMesh->isActive()) {
        commActive = pMesh->getMPICommunicator();
        activeRank = pMesh->getMPIRank();
        activeNpes = pMesh->getMPICommSize();
    } else {
        return;
    }

    globalRank = pMesh->getMPIRankGlobal();
    globalNpes = pMesh->getMPICommSizeGlobal();
    commGlobal = pMesh->getMPIGlobalCommunicator();

    double t_stat;
    double t_stat_g[3];

    const char separator = ' ';
    const int nameWidth = 30;
    const int numWidth = 10;

    char fName[256];
    std::ofstream outfile;

    DendroIntL localSz, globalSz;

    DendroIntL ghostElements;
    DendroIntL localElements;

    DendroIntL ghostNodes;
    DendroIntL localNodes;

    DendroIntL totalSendNode;
    DendroIntL totalRecvNode;

    DendroIntL numCalls;

#ifdef SOLVER_PROFILE_HUMAN_READABLE
    if (!activeRank) {
        sprintf(fName, "%s_im.prof", filePrefix);
        outfile.open(fName, std::fstream::app);
        if (outfile.fail()) {
            std::cout << fName << " file open failed " << std::endl;
            return;
        }

        outfile << "active npes : " << activeNpes << std::endl;
        outfile << "global npes : " << globalNpes << std::endl;
        outfile << "current step : " << currentStep << std::endl;
        outfile << "partition tol : " << dsolve::SOLVER_LOAD_IMB_TOL
                << std::endl;
        outfile << "wavelet tol : " << dsolve::SOLVER_WAVELET_TOL << std::endl;
        outfile << "maxdepth : " << dsolve::SOLVER_MAXDEPTH << std::endl;
    }

    MPI_Comm comm = commActive;
    unsigned int rank = activeRank;

    localSz = pMesh->getNumLocalMeshElements();
    par::Mpi_Reduce(&localSz, &globalSz, 1, MPI_SUM, 0, comm);
    if (!rank) outfile << "Elements : " << globalSz << std::endl;

    localSz = pMesh->getNumLocalMeshNodes();
    par::Mpi_Reduce(&localSz, &globalSz, 1, MPI_SUM, 0, comm);
    if (!rank) outfile << "DOG(zip) : " << globalSz << std::endl;

    localSz = pMesh->getDegOfFreedomUnZip();
    par::Mpi_Reduce(&localSz, &globalSz, 1, MPI_SUM, 0, comm);
    if (!rank) outfile << "DOG(unzip) : " << globalSz << std::endl;

    ghostElements =
        pMesh->getNumPreGhostElements() + pMesh->getNumPostGhostElements();
    localElements = pMesh->getNumLocalMeshElements();

    ghostNodes = pMesh->getNumPreMeshNodes() + pMesh->getNumPostMeshNodes();
    localNodes = pMesh->getNumLocalMeshNodes();

    if (!rank)
        outfile << "========================= MESH "
                   "==========================================================="
                   "============ "
                << std::endl;

    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "step";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "min(#)";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "mean(#)";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "max(#)" << std::endl;

    t_stat = ghostElements;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "ghost Elements";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    t_stat = localElements;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "local Elements";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    t_stat = ghostNodes;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "ghost Nodes";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    t_stat = localNodes;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "local Nodes";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    t_stat = pMesh->getGhostExcgTotalSendNodeCount();
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "send Nodes";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    t_stat = pMesh->getGhostExcgTotalRecvNodeCount();
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "recv Nodes";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    if (!rank)
        outfile << "========================= RUNTIME "
                   "==========================================================="
                   "======== "
                << std::endl;
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "step";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "min(s)";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "mean(s)";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "max(s)" << std::endl;

    /* t_stat=total_runtime.seconds;
    computeOverallStats(&t_stat,t_stat_g,comm);
    if(!rank)outfile << std::left << std::setw(nameWidth) <<
    std::setfill(separator) <<"+runtime(s)"; if(!rank)outfile << std::left <<
    std::setw(nameWidth) << std::setfill(separator)<<t_stat_g[0];
    if(!rank)outfile << std::left << std::setw(nameWidth) <<
    std::setfill(separator)<<t_stat_g[1]; if(!rank)outfile << std::left <<
    std::setw(nameWidth) << std::setfill(separator)<<t_stat_g[2]<<std::endl;


    t_stat=t_f2o.seconds;
    computeOverallStats(&t_stat,t_stat_g,comm);
    if(!rank)outfile << std::left << std::setw(nameWidth) <<
    std::setfill(separator) <<" ++f2o"; if(!rank)outfile << std::left <<
    std::setw(nameWidth) << std::setfill(separator)<<t_stat_g[0];
    if(!rank)outfile << std::left << std::setw(nameWidth) <<
    std::setfill(separator)<<t_stat_g[1]; if(!rank)outfile << std::left <<
    std::setw(nameWidth) << std::setfill(separator)<<t_stat_g[2]<<std::endl;


    t_stat=t_cons.seconds;
    computeOverallStats(&t_stat,t_stat_g,comm);
    if(!rank)outfile << std::left << std::setw(nameWidth) <<
    std::setfill(separator) <<" ++construction"; if(!rank)outfile << std::left
    << std::setw(nameWidth) << std::setfill(separator)<<t_stat_g[0];
    if(!rank)outfile << std::left << std::setw(nameWidth) <<
    std::setfill(separator)<<t_stat_g[1]; if(!rank)outfile << std::left <<
    std::setw(nameWidth) << std::setfill(separator)<<t_stat_g[2]<<std::endl;


    t_stat=t_rkSolve.seconds;
    computeOverallStats(&t_stat,t_stat_g,comm);
    if(!rank)outfile << std::left << std::setw(nameWidth) <<
    std::setfill(separator) <<" ++rkSolve"; if(!rank)outfile << std::left <<
    std::setw(nameWidth) << std::setfill(separator)<<t_stat_g[0];
    if(!rank)outfile << std::left << std::setw(nameWidth) <<
    std::setfill(separator)<<t_stat_g[1]; if(!rank)outfile << std::left <<
    std::setw(nameWidth) << std::setfill(separator)<<t_stat_g[2]<<std::endl;*/

    t_stat = t_bal.snap;
    // numCalls=t_bal.num_calls;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "  ++2:1 balance";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    t_stat = t_mesh.snap;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "  ++mesh";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    t_stat = t_rkStep.snap;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "  ++rkstep";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    t_stat = t_ghostEx_sync.snap;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "  ++ghostExchge.";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    t_stat = t_unzip_sync.snap;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "  ++unzip_sync";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    t_stat = t_unzip_async.snap;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "  ++unzip_async";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS

    t_stat = dendro::timer::t_unzip_async_comm.snap;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "  --unzip_comm_wait (comm) ";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    t_stat = dendro::timer::t_unzip_sync_nodalval.snap;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "  --unzip_nodalVal";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    t_stat = dendro::timer::t_unzip_sync_f_c1.snap;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "  --t_unzip_sync_f_c1";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    t_stat = dendro::timer::t_unzip_sync_f_c2.snap;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "  --t_unzip_sync_f_c2";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    t_stat = dendro::timer::t_unzip_sync_f_c3.snap;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "  --t_unzip_sync_f_c3";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    t_stat = dendro::timer::t_unzip_sync_cpy.snap;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "  --t_unzip_sync_cpy";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    t_stat = dendro::timer::t_unzip_sync_internal.snap;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "  --unzip_sync_internal";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    t_stat = dendro::timer::t_unzip_sync_face[0].snap;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "  --unzip_sync_face_left";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    t_stat = dendro::timer::t_unzip_sync_face[1].snap;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "  --unzip_sync_face_right";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    t_stat = dendro::timer::t_unzip_sync_face[2].snap;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "  --unzip_sync_face_down";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    t_stat = dendro::timer::t_unzip_sync_face[3].snap;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "  --unzip_sync_face_up";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    t_stat = dendro::timer::t_unzip_sync_face[4].snap;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "  --unzip_sync_face_back";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    t_stat = dendro::timer::t_unzip_sync_face[5].snap;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "  --unzip_sync_face_front";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    t_stat = dendro::timer::t_unzip_sync_edge.snap;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "  --unzip_sync_edge";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    t_stat = dendro::timer::t_unzip_sync_vtex.snap;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "  --unzip_sync_vtex";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    t_stat = dendro::timer::t_unzip_p2c.snap;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "  --unzip_p2c";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;
#endif

    /*
    #ifdef ENABLE_DENDRO_PROFILE_COUNTERS
    t_stat=t_unzip_async_internal.snap;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if(!rank)outfile << std::left << std::setw(nameWidth) <<
    std::setfill(separator) <<"  --unzip_internal"; if(!rank)outfile <<
    std::left << std::setw(nameWidth) << std::setfill(separator)<<t_stat_g[0];
    if(!rank)outfile << std::left << std::setw(nameWidth) <<
    std::setfill(separator)<<t_stat_g[1]; if(!rank)outfile << std::left <<
    std::setw(nameWidth) << std::setfill(separator)<<t_stat_g[2]<<std::endl;

    t_stat=t_unzip_async_external.snap;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if(!rank)outfile << std::left << std::setw(nameWidth) <<
    std::setfill(separator) <<"  --unzip_external"; if(!rank)outfile <<
    std::left << std::setw(nameWidth) << std::setfill(separator)<<t_stat_g[0];
    if(!rank)outfile << std::left << std::setw(nameWidth) <<
    std::setfill(separator)<<t_stat_g[1]; if(!rank)outfile << std::left <<
    std::setw(nameWidth) << std::setfill(separator)<<t_stat_g[2]<<std::endl;


    t_stat=t_unzip_async_comm.snap;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if(!rank)outfile << std::left << std::setw(nameWidth) <<
    std::setfill(separator) <<"  --unzip_comm (comm) "; if(!rank)outfile <<
    std::left << std::setw(nameWidth) << std::setfill(separator)<<t_stat_g[0];
    if(!rank)outfile << std::left << std::setw(nameWidth) <<
    std::setfill(separator)<<t_stat_g[1]; if(!rank)outfile << std::left <<
    std::setw(nameWidth) << std::setfill(separator)<<t_stat_g[2]<<std::endl;
    #endif
    */
    t_stat = t_isReMesh.snap;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "  ++isReMesh";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    t_stat = t_gridTransfer.snap;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "  ++gridTransfer";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    t_stat = t_deriv.snap;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "  ++deriv ";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    t_stat = t_rhs.snap;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "  ++compute_rhs ";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    //    t_stat = t_rhs_a.snap;
    //    computeOverallStats(&t_stat, t_stat_g, comm);
    //    if (!rank)
    //        outfile << std::left << std::setw(nameWidth) <<
    //        std::setfill(separator)
    //                << "  --compute_rhs_a ";
    //    if (!rank)
    //        outfile << std::left << std::setw(nameWidth) <<
    //        std::setfill(separator)
    //                << t_stat_g[0];
    //    if (!rank)
    //        outfile << std::left << std::setw(nameWidth) <<
    //        std::setfill(separator)
    //                << t_stat_g[1];
    //    if (!rank)
    //        outfile << std::left << std::setw(nameWidth) <<
    //        std::setfill(separator)
    //                << t_stat_g[2] << std::endl;
    //
    //    t_stat = t_rhs_b.snap;
    //    computeOverallStats(&t_stat, t_stat_g, comm);
    //    if (!rank)
    //        outfile << std::left << std::setw(nameWidth) <<
    //        std::setfill(separator)
    //                << "  --compute_rhs_b ";
    //    if (!rank)
    //        outfile << std::left << std::setw(nameWidth) <<
    //        std::setfill(separator)
    //                << t_stat_g[0];
    //    if (!rank)
    //        outfile << std::left << std::setw(nameWidth) <<
    //        std::setfill(separator)
    //                << t_stat_g[1];
    //    if (!rank)
    //        outfile << std::left << std::setw(nameWidth) <<
    //        std::setfill(separator)
    //                << t_stat_g[2] << std::endl;
    //
    //    t_stat = t_rhs_gt.snap;
    //    computeOverallStats(&t_stat, t_stat_g, comm);
    //    if (!rank)
    //        outfile << std::left << std::setw(nameWidth) <<
    //        std::setfill(separator)
    //                << "  --compute_rhs_gt ";
    //    if (!rank)
    //        outfile << std::left << std::setw(nameWidth) <<
    //        std::setfill(separator)
    //                << t_stat_g[0];
    //    if (!rank)
    //        outfile << std::left << std::setw(nameWidth) <<
    //        std::setfill(separator)
    //                << t_stat_g[1];
    //    if (!rank)
    //        outfile << std::left << std::setw(nameWidth) <<
    //        std::setfill(separator)
    //                << t_stat_g[2] << std::endl;
    //
    //    t_stat = t_rhs_chi.snap;
    //    computeOverallStats(&t_stat, t_stat_g, comm);
    //    if (!rank)
    //        outfile << std::left << std::setw(nameWidth) <<
    //        std::setfill(separator)
    //                << "  --compute_rhs_chi ";
    //    if (!rank)
    //        outfile << std::left << std::setw(nameWidth) <<
    //        std::setfill(separator)
    //                << t_stat_g[0];
    //    if (!rank)
    //        outfile << std::left << std::setw(nameWidth) <<
    //        std::setfill(separator)
    //                << t_stat_g[1];
    //    if (!rank)
    //        outfile << std::left << std::setw(nameWidth) <<
    //        std::setfill(separator)
    //                << t_stat_g[2] << std::endl;
    //
    //    t_stat = t_rhs_At.snap;
    //    computeOverallStats(&t_stat, t_stat_g, comm);
    //    if (!rank)
    //        outfile << std::left << std::setw(nameWidth) <<
    //        std::setfill(separator)
    //                << "  --compute_rhs_At ";
    //    if (!rank)
    //        outfile << std::left << std::setw(nameWidth) <<
    //        std::setfill(separator)
    //                << t_stat_g[0];
    //    if (!rank)
    //        outfile << std::left << std::setw(nameWidth) <<
    //        std::setfill(separator)
    //                << t_stat_g[1];
    //    if (!rank)
    //        outfile << std::left << std::setw(nameWidth) <<
    //        std::setfill(separator)
    //                << t_stat_g[2] << std::endl;
    //
    //    t_stat = t_rhs_K.snap;
    //    computeOverallStats(&t_stat, t_stat_g, comm);
    //    if (!rank)
    //        outfile << std::left << std::setw(nameWidth) <<
    //        std::setfill(separator)
    //                << "  --compute_rhs_K ";
    //    if (!rank)
    //        outfile << std::left << std::setw(nameWidth) <<
    //        std::setfill(separator)
    //                << t_stat_g[0];
    //    if (!rank)
    //        outfile << std::left << std::setw(nameWidth) <<
    //        std::setfill(separator)
    //                << t_stat_g[1];
    //    if (!rank)
    //        outfile << std::left << std::setw(nameWidth) <<
    //        std::setfill(separator)
    //                << t_stat_g[2] << std::endl;
    //
    //    t_stat = t_rhs_Gt.snap;
    //    computeOverallStats(&t_stat, t_stat_g, comm);
    //    if (!rank)
    //        outfile << std::left << std::setw(nameWidth) <<
    //        std::setfill(separator)
    //                << "  --compute_rhs_Gt ";
    //    if (!rank)
    //        outfile << std::left << std::setw(nameWidth) <<
    //        std::setfill(separator)
    //                << t_stat_g[0];
    //    if (!rank)
    //        outfile << std::left << std::setw(nameWidth) <<
    //        std::setfill(separator)
    //                << t_stat_g[1];
    //    if (!rank)
    //        outfile << std::left << std::setw(nameWidth) <<
    //        std::setfill(separator)
    //                << t_stat_g[2] << std::endl;
    //
    //    t_stat = t_rhs_B.snap;
    //    computeOverallStats(&t_stat, t_stat_g, comm);
    //    if (!rank)
    //        outfile << std::left << std::setw(nameWidth) <<
    //        std::setfill(separator)
    //                << "  --compute_rhs_B ";
    //    if (!rank)
    //        outfile << std::left << std::setw(nameWidth) <<
    //        std::setfill(separator)
    //                << t_stat_g[0];
    //    if (!rank)
    //        outfile << std::left << std::setw(nameWidth) <<
    //        std::setfill(separator)
    //                << t_stat_g[1];
    //    if (!rank)
    //        outfile << std::left << std::setw(nameWidth) <<
    //        std::setfill(separator)
    //                << t_stat_g[2] << std::endl;
    //
    t_stat = t_bdyc.snap;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "  ++boundary con ";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    t_stat = t_zip.snap;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "  ++zip";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    t_stat = t_ioVtu.snap;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "  ++vtu";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    t_stat = t_ioCheckPoint.snap;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << "  ++checkpoint";
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[0];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[1];
    if (!rank)
        outfile << std::left << std::setw(nameWidth) << std::setfill(separator)
                << t_stat_g[2] << std::endl;

    if (!rank) outfile.close();
#else

    if (!activeRank) {
        sprintf(fName, "%s_im.prof", filePrefix);
        outfile.open(fName, std::fstream::app);
        if (outfile.fail()) {
            std::cout << fName << " file open failed " << std::endl;
            return;
        }

        // writes the header
        if (currentStep == 0)
            outfile << "step\t act_npes\t glb_npes\t part_tol\t wave_tol\t "
                       "maxdepth\t numOcts\t dof_zip\t dof_unzip\t"
                    << "element_ghost_min\t element_ghost_mean\t "
                       "element_ghost_max\t"
                    << "element_local_min\t element_local_mean\t "
                       "element_local_max\t"
                    << "nodes_local_min\t nodes_local_mean\t nodes_local|max\t"
                    << "send_nodes_min\t send_nodes_mean\t send_nodes_max\t"
                    << "recv_nodes_min\t recv_nodes_mean\t recv_nodes_max\t"
                    << "bal_min\t bal_mean\t bal_max\t"
                    << "mesh_min\t mesh_mean\t mesh_max\t"
                    << "rkstep_min\t rkstep_mean\t rkstep_max\t"
                    << "ghostEx_min\t ghostEx_mean\t ghostEx_max\t"
                    << "unzip_sync_min\t unzip_sync_mean\t unzip_sync_max\t"
                    << "unzip_async_min\t unzip_async_mean\t unzip_async_max\t"
                    << "unzip_async_wait_min\t unzip_async_wait_mean\t "
                       "unzip_async_wait_max\t"
                    << "isRemesh_min\t isRemesh_mean\t isRemesh_max\t"
                    << "GT_min\t GT_mean\t GT_max\t"
                    << "zip_min\t zip_mean\t zip_max\t"
                    << "deriv_min\t deriv_mean\t deriv_max\t"
                    << "rhs_min\t rhs_mean\t rhs_max\t" << std::endl;
    }

    MPI_Comm comm = commActive;
    unsigned int rank = activeRank;

    if (!rank) outfile << currentStep << "\t ";
    if (!rank) outfile << activeNpes << "\t ";
    if (!rank) outfile << globalNpes << "\t ";
    if (!rank) outfile << dsolve::SOLVER_LOAD_IMB_TOL << "\t ";
    if (!rank) outfile << dsolve::SOLVER_WAVELET_TOL << "\t ";
    if (!rank) outfile << dsolve::SOLVER_MAXDEPTH << "\t ";

    localSz = pMesh->getNumLocalMeshElements();
    par::Mpi_Reduce(&localSz, &globalSz, 1, MPI_SUM, 0, comm);
    if (!rank) outfile << globalSz << "\t ";

    localSz = pMesh->getNumLocalMeshNodes();
    par::Mpi_Reduce(&localSz, &globalSz, 1, MPI_SUM, 0, comm);
    if (!rank) outfile << globalSz << "\t ";

    localSz = pMesh->getDegOfFreedomUnZip();
    par::Mpi_Reduce(&localSz, &globalSz, 1, MPI_SUM, 0, comm);
    if (!rank) outfile << globalSz << "\t ";

    ghostElements =
        pMesh->getNumPreGhostElements() + pMesh->getNumPostGhostElements();
    localElements = pMesh->getNumLocalMeshElements();

    t_stat = ghostElements;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << t_stat_g[0] << "\t " << t_stat_g[1] << "\t " << t_stat_g[2]
                << "\t ";

    t_stat = localElements;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << t_stat_g[0] << "\t " << t_stat_g[1] << "\t " << t_stat_g[2]
                << "\t ";

    ghostNodes = pMesh->getNumPreMeshNodes() + pMesh->getNumPostMeshNodes();
    localNodes = pMesh->getNumLocalMeshNodes();

    /*t_stat=ghostNodes;
    computeOverallStats(&t_stat,t_stat_g,comm);
    if(!rank) outfile<<t_stat_g[0]<<"\t "<<t_stat_g[1]<<"\t "<<t_stat_g[2]<<"\t
    ";*/

    t_stat = localNodes;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << t_stat_g[0] << "\t " << t_stat_g[1] << "\t " << t_stat_g[2]
                << "\t ";

    t_stat = pMesh->getGhostExcgTotalSendNodeCount();
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << t_stat_g[0] << "\t " << t_stat_g[1] << "\t " << t_stat_g[2]
                << "\t ";

    t_stat = pMesh->getGhostExcgTotalRecvNodeCount();
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << t_stat_g[0] << "\t " << t_stat_g[1] << "\t " << t_stat_g[2]
                << "\t ";

    t_stat = t_bal.snap;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << t_stat_g[0] << "\t " << t_stat_g[1] << "\t " << t_stat_g[2]
                << "\t ";

    t_stat = t_mesh.snap;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << t_stat_g[0] << "\t " << t_stat_g[1] << "\t " << t_stat_g[2]
                << "\t ";

    t_stat = t_rkStep.snap;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << t_stat_g[0] << "\t " << t_stat_g[1] << "\t " << t_stat_g[2]
                << "\t ";

    t_stat = t_ghostEx_sync.snap;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << t_stat_g[0] << "\t " << t_stat_g[1] << "\t " << t_stat_g[2]
                << "\t ";

    t_stat = t_unzip_sync.snap;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << t_stat_g[0] << "\t " << t_stat_g[1] << "\t " << t_stat_g[2]
                << "\t ";

    t_stat = t_unzip_async.snap;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << t_stat_g[0] << "\t " << t_stat_g[1] << "\t " << t_stat_g[2]
                << "\t ";

    t_stat = dendro::timer::t_unzip_async_comm.snap;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << t_stat_g[0] << "\t " << t_stat_g[1] << "\t " << t_stat_g[2]
                << "\t ";

    t_stat = t_isReMesh.snap;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << t_stat_g[0] << "\t " << t_stat_g[1] << "\t " << t_stat_g[2]
                << "\t ";

    t_stat = t_gridTransfer.snap;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << t_stat_g[0] << "\t " << t_stat_g[1] << "\t " << t_stat_g[2]
                << "\t ";

    t_stat = t_deriv.snap;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << t_stat_g[0] << "\t " << t_stat_g[1] << "\t " << t_stat_g[2]
                << "\t ";

    t_stat = t_rhs.snap;
    computeOverallStats(&t_stat, t_stat_g, comm);
    if (!rank)
        outfile << t_stat_g[0] << "\t " << t_stat_g[1] << "\t " << t_stat_g[2]
                << "\t ";

    if (!rank) outfile << std::endl;
    if (!rank) outfile.close();
#endif
}

}  // namespace timer

}  // namespace dsolve
