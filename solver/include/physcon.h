#ifndef PHYSCON_H
#define PHYSCON_H

#include <iostream>

#include "derivs.h"
#include "grUtils.h"
#include "parameters.h"

void physical_constraints(double **uZipConVars, const double **uZipVars,
                          const unsigned int &offset, const double *pmin,
                          const double *pmax, const unsigned int *sz,
                          const unsigned int &bflag);

void physical_constraints_compact_derivs(double **uZipConVars,
                                         double **uZipVars,
                                         const unsigned int &offset,
                                         const double *pmin, const double *pmax,
                                         const unsigned int *sz,
                                         const unsigned int &bflag);

#endif
