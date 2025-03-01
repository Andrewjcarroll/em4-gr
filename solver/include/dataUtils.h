//
// Created by David
//
/**
 * @author Milinda Fernando / David Van Komen
 * School of Computing, University of Utah
 * @brief Contins utility functions for processing simulation data and post
 * processing
 */

#ifndef DENDRO_5_0_DATAUTILS_H
#define DENDRO_5_0_DATAUTILS_H

#include "TreeNode.h"
#include "grDef.h"
#include "mesh.h"
#include "parameters.h"
#include "point.h"

namespace dsolve {

bool isReMeshWAMR(
    ot::Mesh *pMesh, const double **unzippedVec, const unsigned int *varIds,
    const unsigned int numVars,
    std::function<double(double, double, double, double *)> wavelet_tol,
    double amr_coarse_fac);

/**
 * @brief refine ratially based on the BH locations and AMR_R.
 * @param pMesh pointer to the mesh object.
 */
// bool isReMeshBHRadial(ot::Mesh* pMesh);

}  // end of namespace dsolve

#endif  // DENDRO_5_0_DATAUTILS_H
