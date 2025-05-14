#ifndef system_constraints_H
#define system_constraints_H

#include <iostream>

#include "grDef.h"
#include "grUtils.h"
#include "parameters.h"

using namespace dsolve;

/*----------------------------------------------------------------------;
 *
 * enforce physical constraints on SOLVER variables:
 *            det(gt) = 1,  tr(At) = 0,  alpha > 0 and chi >0.
 *
 *
 * DFVK NOTE: i'm not sure if this is entirely "constraint" since it seems to
 * just be trying to make sure that the metric determinent is positive for the
 * gtd while also keeping the Atd variable clean. Either way, if there are
 * special constraints like this, we need to generate this code
 *----------------------------------------------------------------------*/
inline void enforce_system_constraints(double **uiVar,
                                       const unsigned int node) {
//If the Divergence of B gets too large we will kill the run to save resources:
double Divergence_B = uiVar[VAR_CONSTRAINT::C_DIVB][node];
if(abs(Divergence_B)>10.0){
    std::cout <<"The divergence has gotten too large DIVB = "<<Divergence_B << std::endl;
    exit(0);
}
}

#endif
