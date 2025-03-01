//
// Created by milinda on 1/16/19.
//

#include "dataUtils.h"

namespace dsolve {


bool isReMeshWAMR(
    ot::Mesh *pMesh, const double **unzippedVec, const unsigned int *varIds,
    const unsigned int numVars,
    std::function<double(double, double, double, double *)> wavelet_tol,
    double amr_coarse_fac) {
    // if (!(pMesh->isReMeshUnzip((const double **)unzippedVec, varIds, numVars,
    //                            wavelet_tol, dsolve::SOLVER_DENDRO_AMR_FAC)))
    //     return false;

    std::vector<unsigned int> refine_flags;


    const unsigned int eleLocalBegin = pMesh->getElementLocalBegin();
    const unsigned int eleLocalEnd = pMesh->getElementLocalEnd();
    bool isOctChange = false;
    bool isOctChange_g = false;
    Point d1, d2, temp;

    const unsigned int eOrder = pMesh->getElementOrder();




    if (pMesh->isActive()) {


        const RefElement *refEl = pMesh->getReferenceElement();
        wavelet::WaveletEl *wrefEl =
            new wavelet::WaveletEl((RefElement *)refEl);

        refine_flags.resize(pMesh->getNumLocalMeshElements(), OCT_NO_CHANGE);
        const ot::TreeNode *pNodes = pMesh->getAllElements().data();

        std::vector<double> wtol_vals;
        wtol_vals.resize(SOLVER_NUM_VARS, 0);

        const std::vector<ot::Block> &blkList = pMesh->getLocalBlockList();
        const unsigned int eOrder = pMesh->getElementOrder();

        const unsigned int nx = (2 * eOrder + 1);
        const unsigned int ny = (2 * eOrder + 1);
        const unsigned int nz = (2 * eOrder + 1);

        const unsigned int sz_per_dof = nx * ny * nz;
        const unsigned int isz[] = {nx, ny, nz};
        std::vector<double> eVecTmp;
        eVecTmp.resize(sz_per_dof);

        std::vector<double> wCout;
        wCout.resize(sz_per_dof);

        for (unsigned int blk = 0; blk < blkList.size(); blk++) {
            const unsigned int pw = blkList[blk].get1DPadWidth();
            const unsigned int bflag = blkList[blk].getBlkNodeFlag();
            assert(pw == (eOrder >> 1u));

            for (unsigned int ele = blkList[blk].getLocalElementBegin();
                 ele < blkList[blk].getLocalElementEnd(); ele++) {
                const bool isBdyOct = pMesh->isBoundaryOctant(ele);
                const double oct_dx =
                    (1u << (m_uiMaxDepth - pNodes[ele].getLevel())) /
                    (double(eOrder));

                Point oct_pt1 = Point(pNodes[ele].minX(), pNodes[ele].minY(),
                                      pNodes[ele].minZ());
                Point oct_pt2 = Point(pNodes[ele].minX() + oct_dx,
                                      pNodes[ele].minY() + oct_dx,
                                      pNodes[ele].minZ() + oct_dx);
                Point domain_pt1, domain_pt2, dx_domain;
                pMesh->octCoordToDomainCoord(oct_pt1, domain_pt1);
                pMesh->octCoordToDomainCoord(oct_pt2, domain_pt2);
                dx_domain = domain_pt2 - domain_pt1;
                double hx[3] = {dx_domain.x(), dx_domain.y(), dx_domain.z()};
                const double tol_ele = wavelet_tol(
                    domain_pt1.x(), domain_pt1.y(), domain_pt1.z(), hx);

                // initialize all the wavelet errors to zero initially.
                for (unsigned int v = 0; v < SOLVER_NUM_VARS; v++)
                    wtol_vals[v] = 0;

                for (unsigned int v = 0; v < numVars; v++) {
                    const unsigned int vid = varIds[v];
                    pMesh->getUnzipElementalNodalValues(
                        unzippedVec[vid], blk, ele, eVecTmp.data(), true);

                    // computes the wavelets.
                    wrefEl->compute_wavelets_3D((double *)(eVecTmp.data()), isz,
                                                wCout, isBdyOct);
                    wtol_vals[vid] = (normL2(wCout.data(), wCout.size())) /
                                     sqrt(wCout.size());

                    // early bail if the computed tolerance valule is large.
                    if (wtol_vals[vid] > tol_ele) break;
                }

                const double l_max = vecMax(wtol_vals.data(), wtol_vals.size());
                if (l_max > tol_ele) {
                    refine_flags[(ele - eleLocalBegin)] = OCT_SPLIT;
                } else if (l_max < amr_coarse_fac * tol_ele) {
                    refine_flags[(ele - eleLocalBegin)] = OCT_COARSE;
                } else {
                    refine_flags[(ele - eleLocalBegin)] = OCT_NO_CHANGE;
                }
            }
        }

        delete wrefEl;

        // --- Below code enforces the artifical refinement by looking at the
        // puncture locations, by
        // --- overiding what currently set by the wavelets.
        for (unsigned int ele = eleLocalBegin; ele < eleLocalEnd; ele++) {
            // refine_flags[ele-eleLocalBegin] =
            // (pNodes[ele].getFlag()>>NUM_LEVEL_BITS); std::cout<<"ref flag:
            // "<<(pNodes[ele].getFlag()>>NUM_LEVEL_BITS)<<std::endl;
            // if(refine_flags[ele-eleLocalBegin]==OCT_SPLIT)
            pMesh->octCoordToDomainCoord(
                Point((double)pNodes[ele].minX(), (double)pNodes[ele].minY(),
                      (double)pNodes[ele].minZ()),
                temp);

            //@milinda: 11/21/2020 : Don't allow to violate the min depth
            if (pNodes[ele].getLevel() < dsolve::SOLVER_MINDEPTH) {
                refine_flags[ele - eleLocalBegin] = OCT_SPLIT;
            } else if (pNodes[ele].getLevel() == dsolve::SOLVER_MINDEPTH &&
                       refine_flags[ele - eleLocalBegin] == OCT_COARSE) {
                refine_flags[ele - eleLocalBegin] = OCT_NO_CHANGE;
            }

        }

        isOctChange = pMesh->setMeshRefinementFlags(refine_flags);
    }

    MPI_Allreduce(&isOctChange, &isOctChange_g, 1, MPI_CXX_BOOL, MPI_LOR,
                  pMesh->getMPIGlobalCommunicator());
    return isOctChange_g;
}

// ignoring remesh bh radial, in BSSN code, but Milinda says it's too expensive
// to use

}  // end of namespace dsolve
