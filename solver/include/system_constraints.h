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
    // clang-format off
    /*[[[cog
    import cog
    import sys
    import importlib.util
    import dendrosym

    cog.outl('// clang-format on')

    # the following lines will import any module directly from
    spec = importlib.util.spec_from_file_location("dendroconf", CONFIG_FILE_PATH)
    dendroconf = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = dendroconf
    spec.loader.exec_module(dendroconf)

    cog.outl('//// SOLVER CONSTRAINTS')
    cog.outl(dendroconf.dendroConfigs.generate_evolution_constraints())

    ]]]*/
    // clang-format on
    //[[[end]]]
}

#endif
