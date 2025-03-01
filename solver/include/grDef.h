//
// Created by milinda on 10/8/18.
//

#ifndef DENDRO_5_0_GRDEF_H
#define DENDRO_5_0_GRDEF_H

#define Rx (dsolve::SOLVER_COMPD_MAX[0] - dsolve::SOLVER_COMPD_MIN[0])
#define Ry (dsolve::SOLVER_COMPD_MAX[1] - dsolve::SOLVER_COMPD_MIN[1])
#define Rz (dsolve::SOLVER_COMPD_MAX[2] - dsolve::SOLVER_COMPD_MIN[2])

#define RgX (dsolve::SOLVER_OCTREE_MAX[0] - dsolve::SOLVER_OCTREE_MIN[0])
#define RgY (dsolve::SOLVER_OCTREE_MAX[1] - dsolve::SOLVER_OCTREE_MIN[1])
#define RgZ (dsolve::SOLVER_OCTREE_MAX[2] - dsolve::SOLVER_OCTREE_MIN[2])

#define GRIDX_TO_X(xg)                                    \
    (((Rx / RgX) * (xg - dsolve::SOLVER_OCTREE_MIN[0])) + \
     dsolve::SOLVER_COMPD_MIN[0])
#define GRIDY_TO_Y(yg)                                    \
    (((Ry / RgY) * (yg - dsolve::SOLVER_OCTREE_MIN[1])) + \
     dsolve::SOLVER_COMPD_MIN[1])
#define GRIDZ_TO_Z(zg)                                    \
    (((Rz / RgZ) * (zg - dsolve::SOLVER_OCTREE_MIN[2])) + \
     dsolve::SOLVER_COMPD_MIN[2])

#define X_TO_GRIDX(xc)                                   \
    (((RgX / Rx) * (xc - dsolve::SOLVER_COMPD_MIN[0])) + \
     dsolve::SOLVER_OCTREE_MIN[0])
#define Y_TO_GRIDY(yc)                                   \
    (((RgY / Ry) * (yc - dsolve::SOLVER_COMPD_MIN[1])) + \
     dsolve::SOLVER_OCTREE_MIN[1])
#define Z_TO_GRIDZ(zc)                                   \
    (((RgZ / Rz) * (zc - dsolve::SOLVER_COMPD_MIN[2])) + \
     dsolve::SOLVER_OCTREE_MIN[2])

// type of the rk method.
enum RKType { RK3 = 0, RK4, RK45 };

namespace dsolve {
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

cog.outl(dendroconf.dendroConfigs.gen_enum_code("evolution"))

cog.outl(dendroconf.dendroConfigs.gen_enum_code("constraint", enum_name="VAR_CONSTRAINT"))

]]]*/
// clang-format on
enum VAR {
    U_E0 = 0,
    U_E1,
    U_E2,
    U_B0,
    U_B1,
    U_B2,
    U_PHI,
    U_PSI,
};
enum VAR_CONSTRAINT {
    C_DIVE = 0,
    C_DIVB,
};
//[[[end]]]

// declare the enum names for human-readable printing later
// clang-format off
/*[[[cog
cog.outl('// clang-format on')
cog.outl(dendroconf.dendroConfigs.gen_enum_names("evolution"))
cog.outl(dendroconf.dendroConfigs.gen_enum_names("constraint", enum_name="VAR_CONSTRAINT"))
cog.outl(dendroconf.dendroConfigs.gen_enum_iterable_list("evolution"))
cog.outl(dendroconf.dendroConfigs.gen_enum_iterable_list("constraint", enum_name="VAR_CONSTRAINT"))
]]]*/
// clang-format on
static const char *SOLVER_VAR_NAMES[] = {

"U_E0","U_E1","U_E2","U_B0","U_B1","U_B2","U_PHI","U_PSI"
};

static const char *SOLVER_VAR_CONSTRAINT_NAMES[] = {
    "C_DIVE",
    "C_DIVB",
};

static const VAR SOLVER_VAR_ITERABLE_LIST[] = {
    U_E0, U_E1, U_E2, U_B0, U_B1, U_B2, U_PHI, U_PSI,
};

static const VAR_CONSTRAINT SOLVER_VAR_CONSTRAINT_ITERABLE_LIST[] = {
    C_DIVE,
    C_DIVB,
};

//[[[end]]]

/**
 * @brief Refinement mode types.
 * WAMR : Wavelet based refinement.
 * REFINE_MODE_NONE : no refinement ever!
 * REFINEMENT_END : only used for bound checking the refinment mode, doesn't
 * imply a real mode
 */
enum RefinementMode { WAMR = 0, REFINE_MODE_NONE, REFINEMENT_END };

}  // end of namespace dsolve

#endif  // DENDRO_5_0_GRDEF_H
