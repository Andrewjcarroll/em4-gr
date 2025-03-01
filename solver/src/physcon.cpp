#include "physcon.h"

#include "debugger_tools.h"
#include "solver_main.h"

// TEMPORARY
// #ifdef SOLVER_DEBUG_FLOATING_POINT
// #pragma message("Enabled debugging floating point errors")

#define PI 3.14159265358979323846

#include <fenv.h>

// #endif

void show_fe_exceptions(void) {
    printf("exceptions raised:");
    if (fetestexcept(FE_DIVBYZERO)) printf(" FE_DIVBYZERO");
    if (fetestexcept(FE_INEXACT)) printf(" FE_INEXACT");
    if (fetestexcept(FE_INVALID)) printf(" FE_INVALID");
    if (fetestexcept(FE_OVERFLOW)) printf(" FE_OVERFLOW");
    if (fetestexcept(FE_UNDERFLOW)) printf(" FE_UNDERFLOW");
    feclearexcept(FE_ALL_EXCEPT);
    printf("\n");
}
// END TEMP

using namespace dsolve;

/*----------------------------------------------------------------------
 *
 * vector form of RHS
 *
 *----------------------------------------------------------------------*/
void physical_constraints(double **uZipConVars, const double **uZipVars,
                          const unsigned int &offset, const double *pmin,
                          const double *pmax, const unsigned int *sz,
                          const unsigned int &bflag) {
    // wait_for_debugger();

    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];
    const unsigned int n = nx * ny * nz;
    unsigned int BLK_SZ = n;

    // declare the size of bytes for memory allocation down the line
    const unsigned int bytes = n * sizeof(double);

    const double hx = (pmax[0] - pmin[0]) / (nx - 1);
    const double hy = (pmax[1] - pmin[1]) / (ny - 1);
    const double hz = (pmax[2] - pmin[2]) / (nz - 1);

    // printf("ALL EXCEPTIONS BEFORE COMPUTING CONSTRAINTS: ");
    // show_fe_exceptions();

    // HAM is hamiltonian constraint
    // mom is 3 component vector - momentum constraints

    // clang-format off
    /*[[[cog
    import cog
    import sys
    import os
    import importlib.util
    import dendrosym

    cog.outl('// clang-format on')

    # get the current working directory, should be root of project
    current_path = os.getcwd()
    output_path = os.path.join(current_path, "gencode")

    # the following lines will import any module directly from
    spec = importlib.util.spec_from_file_location("dendroconf", CONFIG_FILE_PATH)
    dendroconf = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = dendroconf
    spec.loader.exec_module(dendroconf)

    cog.outl('//// GENERATED CONSTRAINT VARIABLES')
    cog.outl(dendroconf.dendroConfigs.generate_variable_extraction(
        "constraint", use_const=False, zip_var_name="uZipConVars"
    ))

    cog.outl('/// GENERATED CONSTANT EVOLUTION VARIABLES')
    cog.outl(dendroconf.dendroConfigs.generate_variable_extraction(
        "evolution", use_const=True, zip_var_name="uZipVars"
    ))
    ]]]*/
    // clang-format on

    // ADDED PHYSCON FROM EM3 CODE -AJC
    double *divE = &uZipConVars[VAR_CONSTRAINT::C_DIVE][offset];
    double *divB = &uZipConVars[VAR_CONSTRAINT::C_DIVB][offset];

    const double *E0 = &uZipVars[VAR::U_E0][offset];
    const double *E1 = &uZipVars[VAR::U_E1][offset];
    const double *E2 = &uZipVars[VAR::U_E2][offset];

    const double *B0 = &uZipVars[VAR::U_B0][offset];
    const double *B1 = &uZipVars[VAR::U_B1][offset];
    const double *B2 = &uZipVars[VAR::U_B2][offset];
    const double *Phi = &uZipVars[VAR::U_PHI][offset];
    const double *Psi = &uZipVars[VAR::U_PSI][offset];
    //[[[end]]]

    const unsigned int PW = dsolve::SOLVER_PADDING_WIDTH;

    // clang-format off
    /*[[[cog
    cog.outl('// clang-format on')
    cog.outl("// PARAMETER EXTRACTION FOR CONSTRAINTS")

    cog.outl(dendroconf.dendroConfigs.gen_parameter_code("constraint"))

    ]]]*/
    // clang-format on
    // PARAMETER EXTRACTION FOR CONSTRAINTS

    //[[[end]]]

    mem::memory_pool<double> *__mem_pool = &SOLVER_MEM_POOL;

    // get derivative workspace
    double *const deriv_base = dsolve::SOLVER_DERIV_WORKSPACE;

    // clang-format off
    /*[[[cog
    cog.outl('// clang-format on')
    
    cog.outl("//GENERATED ADVANCED DERIVATIVE EQUATIONS")

    print("Now generating advanced derivatves for physcon", file=sys.stderr)
    
    # note that we need to store the deallocation string as well for later down the line!
    (intermediate_grad_str, 
     deallocate_intermediate_grad_str) = dendroconf.dendroConfigs.generate_pre_necessary_derivatives(
         "constraint", dtype="double"
     )

    print("Finished generating advanced derivatves", file=sys.stderr)

    intermediate_filename = "solver_physcon_intermediate_grad.cpp.inc"

    with open(os.path.join(output_path, intermediate_filename), "w") as f:
        f.write(intermediate_grad_str)

    print("Saved them to file", file=sys.stderr)

    cog.outl(f'#include "../gencode/{intermediate_filename}"')
    
    ]]]*/

    //[[[end]]]

    // printf("ALL EXCEPTIONS AFTER COMPUTING INTERMEDIATE GRADIENT: ");
    // show_fe_exceptions();

    // create the files that have the derivative memory allocations and
    // calculations
    // clang-format off
    /*[[[cog
    cog.outl('// clang-format on')

    deriv_alloc, deriv_calc, deriv_dealloc = dendroconf.dendroConfigs.generate_deriv_allocation_and_calc("constraint")

    print("Generated derivative allocation, calculation, and deallocation code for Evolution", file=sys.stderr)

    alloc_filename = "solver_physcon_deriv_memalloc.cpp.inc"

    with open(os.path.join(output_path, alloc_filename), "w") as f:
        f.write(deriv_alloc)

    cog.outl(f'#include "../gencode/{alloc_filename}"')

    calc_filename = "solver_physcon_deriv_calc.cpp.inc"

    with open(os.path.join(output_path, calc_filename), "w") as f:
        f.write(deriv_calc)

    cog.outl(f'#include "../gencode/{calc_filename}"')

    dealloc_filename = "solver_physcon_deriv_memdealloc.cpp.inc"

    with open(os.path.join(output_path, dealloc_filename), "w") as f:
        f.write(deriv_dealloc)

    ]]]*/

#include "../gencode/solver_physcon_deriv_memalloc.cpp.inc"
#include "../gencode/solver_physcon_deriv_calc.cpp.inc"
    // clang-format on
    //[[[end]]]
    double *rho_e = deriv_base + 9 * BLK_SZ;

    // TODO: Why is Rho_e set to 0? Is this supposed to do something different?
    for (unsigned int m = 0; m < n; m++) {
        rho_e[m] = 0.0;
    }

    // printf("ALL EXCEPTIONS AFTER COMPUTING DERIVATIVES: ");
    // show_fe_exceptions();

    // enforce hamiltonian and momentum constraints
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

                // clang-format off
                /*[[[cog

                cog.outl('// clang-format on')

                physcon_rhs_code = dendroconf.dendroConfigs.generate_rhs_code("constraint", include_rhs_in_name=False)
                physcon_filename = "solver_physcon_eqns.cpp.inc"

                with open(os.path.join(output_path, physcon_filename), "w") as f:
                    f.write(physcon_rhs_code)
                
                cog.outl(f'#include "../gencode/{physcon_filename}"')
                
                ]]]*/
                #include "../gencode/solver_physcon_eqns.cpp.inc"
                //[[[end]]]

                // clang-format on
            }
        }
    }

    // clang-format off
    /*[[[cog

    cog.outl('// clang-format on')

    cog.outl(deallocate_intermediate_grad_str)

    cog.outl(f'#include "../gencode/{dealloc_filename}"')
    

    ]]]*/
#include "../gencode/solver_physcon_deriv_memdealloc.cpp.inc"
    //[[[end]]]
    // clang-format on
}

void physical_constraints_compact_derivs(double **uZipConVars,
                                         double **uZipVars,
                                         const unsigned int &offset,
                                         const double *pmin, const double *pmax,
                                         const unsigned int *sz,
                                         const unsigned int &bflag) {
    // wait_for_debugger();

    using namespace dendro_cfd;

    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];
    const unsigned int n = nx * ny * nz;
    unsigned int BLK_SZ = n;

    // declare the size of bytes for memory allocation down the line
    const unsigned int bytes = n * sizeof(double);

    const double hx = (pmax[0] - pmin[0]) / (nx - 1);
    const double hy = (pmax[1] - pmin[1]) / (ny - 1);
    const double hz = (pmax[2] - pmin[2]) / (nz - 1);

    // printf("ALL EXCEPTIONS BEFORE COMPUTING CONSTRAINTS: ");
    // show_fe_exceptions();

    // HAM is hamiltonian constraint
    // mom is 3 component vector - momentum constraints

    // clang-format off
    /*[[[cog
    import cog
    import sys
    import os
    import importlib.util
    import dendrosym

    cog.outl('// clang-format on')

    # get the current working directory, should be root of project
    current_path = os.getcwd()
    output_path = os.path.join(current_path, "gencode")

    # the following lines will import any module directly from
    spec = importlib.util.spec_from_file_location("dendroconf", CONFIG_FILE_PATH)
    dendroconf = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = dendroconf
    spec.loader.exec_module(dendroconf)

    cog.outl('//// GENERATED CONSTRAINT VARIABLES')
    cog.outl(dendroconf.dendroConfigs.generate_variable_extraction(
        "constraint", use_const=False, zip_var_name="uZipConVars"
    ))

    cog.outl('/// GENERATED CONSTANT EVOLUTION VARIABLES')
    cog.outl(dendroconf.dendroConfigs.generate_variable_extraction(
        "evolution", use_const=True, zip_var_name="uZipVars"
    ))
    ]]]*/
    // clang-format on

    // ADDED PHYSCON FROM EM3 CODE -AJC
    double *divE = &uZipConVars[VAR_CONSTRAINT::C_DIVE][offset];
    double *divB = &uZipConVars[VAR_CONSTRAINT::C_DIVB][offset];

    double *E0 = &uZipVars[VAR::U_E0][offset];
    double *E1 = &uZipVars[VAR::U_E1][offset];
    double *E2 = &uZipVars[VAR::U_E2][offset];

    double *B0 = &uZipVars[VAR::U_B0][offset];
    double *B1 = &uZipVars[VAR::U_B1][offset];
    double *B2 = &uZipVars[VAR::U_B2][offset];

    double *Phi = &uZipVars[VAR::U_PHI][offset];
    double *Psi = &uZipVars[VAR::U_PSI][offset];

    //[[[end]]]

    const unsigned int PW = dsolve::SOLVER_PADDING_WIDTH;

#ifndef DENDRO_USE_NEW_DERIVS
    if (bflag != 0) {
        cfd.clear_boundary_padding_nans(E0, sz, bflag);
        cfd.clear_boundary_padding_nans(E1, sz, bflag);
        cfd.clear_boundary_padding_nans(E2, sz, bflag);

        cfd.clear_boundary_padding_nans(B0, sz, bflag);
        cfd.clear_boundary_padding_nans(B1, sz, bflag);
        cfd.clear_boundary_padding_nans(B2, sz, bflag);

        cfd.clear_boundary_padding_nans(Phi, sz, bflag);
        cfd.clear_boundary_padding_nans(Psi, sz, bflag);
    }
#endif

    // clang-format off
    /*[[[cog
    cog.outl('// clang-format on')
    cog.outl("// PARAMETER EXTRACTION FOR CONSTRAINTS")

    cog.outl(dendroconf.dendroConfigs.gen_parameter_code("constraint"))

    ]]]*/
    // clang-format on
    // PARAMETER EXTRACTION FOR CONSTRAINTS

    //[[[end]]]

    mem::memory_pool<double> *__mem_pool = &SOLVER_MEM_POOL;

    // get derivative workspace
    double *const deriv_base = dsolve::SOLVER_DERIV_WORKSPACE;

    // clang-format off
    /*[[[cog
    cog.outl('// clang-format on')
    
    cog.outl("//GENERATED ADVANCED DERIVATIVE EQUATIONS")

    print("Now generating advanced derivatves for physcon", file=sys.stderr)
    
    # note that we need to store the deallocation string as well for later down the line!
    (intermediate_grad_str, 
     deallocate_intermediate_grad_str) = dendroconf.dendroConfigs.generate_pre_necessary_derivatives(
         "constraint", dtype="double"
     )

    print("Finished generating advanced derivatves", file=sys.stderr)

    intermediate_filename = "solver_physcon_intermediate_grad.cpp.inc"

    with open(os.path.join(output_path, intermediate_filename), "w") as f:
        f.write(intermediate_grad_str)

    print("Saved them to file", file=sys.stderr)

    cog.outl(f'#include "../gencode/{intermediate_filename}"')
    
    ]]]*/

    //[[[end]]]

    // printf("ALL EXCEPTIONS AFTER COMPUTING INTERMEDIATE GRADIENT: ");
    // show_fe_exceptions();

    // create the files that have the derivative memory allocations and
    // calculations
    // clang-format off
    /*[[[cog
    cog.outl('// clang-format on')

    deriv_alloc, deriv_calc, deriv_dealloc = dendroconf.dendroConfigs.generate_deriv_allocation_and_calc("constraint")

    print("Generated derivative allocation, calculation, and deallocation code for Evolution", file=sys.stderr)

    alloc_filename = "solver_physcon_deriv_memalloc.cpp.inc"

    with open(os.path.join(output_path, alloc_filename), "w") as f:
        f.write(deriv_alloc)

    cog.outl(f'#include "../gencode/{alloc_filename}"')

    calc_filename = "solver_physcon_deriv_calc.cpp.inc"

    with open(os.path.join(output_path, calc_filename), "w") as f:
        f.write(deriv_calc)

    cog.outl(f'#include "../gencode/{calc_filename}"')

    dealloc_filename = "solver_physcon_deriv_memdealloc.cpp.inc"

    with open(os.path.join(output_path, dealloc_filename), "w") as f:
        f.write(deriv_dealloc)

    ]]]*/

#include "../gencode/solver_physcon_deriv_memalloc.cpp.inc"
// #include "../gencode/solver_physcon_deriv_calc.cpp.inc"
    // clang-format on
    //[[[end]]]
    //

#ifdef DENDRO_USE_NEW_DERIVS

    SOLVER_DERIVS->grad_x(grad_0_E0, E0, hx, sz, bflag);
    SOLVER_DERIVS->grad_y(grad_1_E1, E1, hy, sz, bflag);
    SOLVER_DERIVS->grad_z(grad_2_E2, E2, hz, sz, bflag);

    SOLVER_DERIVS->grad_x(grad_0_B0, B0, hx, sz, bflag);
    SOLVER_DERIVS->grad_y(grad_1_B1, B1, hy, sz, bflag);
    SOLVER_DERIVS->grad_z(grad_2_B2, B2, hz, sz, bflag);

#else
    if (dsolve::SOLVER_DERIV_TYPE == dendro_cfd::CFD_NONE) {
        dendro_derivs::deriv_x(grad_0_E0, E0, hx, sz, bflag);
        dendro_derivs::deriv_y(grad_1_E1, E1, hy, sz, bflag);
        dendro_derivs::deriv_z(grad_2_E2, E2, hz, sz, bflag);

        dendro_derivs::deriv_x(grad_0_B0, B0, hx, sz, bflag);
        dendro_derivs::deriv_y(grad_1_B1, B1, hy, sz, bflag);
        dendro_derivs::deriv_z(grad_2_B2, B2, hz, sz, bflag);
    } else {
        cfd.cfd_x(grad_0_E0, E0, hx, sz, bflag);
        cfd.cfd_y(grad_1_E1, E1, hy, sz, bflag);
        cfd.cfd_z(grad_2_E2, E2, hz, sz, bflag);

        cfd.cfd_x(grad_0_B0, B0, hx, sz, bflag);
        cfd.cfd_y(grad_1_B1, B1, hy, sz, bflag);
        cfd.cfd_z(grad_2_B2, B2, hz, sz, bflag);
    }
#endif

    double *rho_e = deriv_base + 9 * BLK_SZ;

    // TODO: Why is Rho_e set to 0? Is this supposed to do something different?
    for (unsigned int m = 0; m < n; m++) {
        rho_e[m] = 0.0;
    }

    // printf("ALL EXCEPTIONS AFTER COMPUTING DERIVATIVES: ");
    // show_fe_exceptions();

    // enforce hamiltonian and momentum constraints
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

                // clang-format off
                /*[[[cog

                cog.outl('// clang-format on')

                physcon_rhs_code = dendroconf.dendroConfigs.generate_rhs_code("constraint", include_rhs_in_name=False)
                physcon_filename = "solver_physcon_eqns.cpp.inc"

                with open(os.path.join(output_path, physcon_filename), "w") as f:
                    f.write(physcon_rhs_code)
                
                cog.outl(f'#include "../gencode/{physcon_filename}"')
                
                ]]]*/
                #include "../gencode/solver_physcon_eqns.cpp.inc"
                //[[[end]]]

                // clang-format on
            }
        }
    }

    // clang-format off
    /*[[[cog

    cog.outl('// clang-format on')

    cog.outl(deallocate_intermediate_grad_str)

    cog.outl(f'#include "../gencode/{dealloc_filename}"')
    

    ]]]*/
#include "../gencode/solver_physcon_deriv_memdealloc.cpp.inc"
    //[[[end]]]
    // clang-format on
}
