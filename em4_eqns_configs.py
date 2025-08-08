import dendrosym
import sympy as sym

pi = sym.pi
exp = sym.exp
pow = sym.Pow
sqrt = sym.sqrt
Rational = sym.Rational

# DEFINE: dendro config class to use for generating code
dendroConfigs = dendrosym.NRConfig("em4")

# the indexing used for the dendro configs
idx_str = "[pp]"

# save the index string
dendroConfigs.set_idx_str(idx_str)

# RHS variables
Psi = dendrosym.dtypes.scalar("Psi" + idx_str)
Phi = dendrosym.dtypes.scalar("Phi" + idx_str)

B = dendrosym.dtypes.vec3("B" + idx_str)
E = dendrosym.dtypes.vec3("E" + idx_str)

# add in the evolution variables
dendroConfigs.add_evolution_variables([Psi, Phi, B, E])

# ----- PARAMETER VARIABLES
# kappa_1 and kappa_2 are parameter variables
kappa_1_param = dendrosym.dtypes.ParameterVariable(
    "kappa_1", dtype="double", num_params=1
)
kappa_2_param = dendrosym.dtypes.ParameterVariable(
    "kappa_2", dtype="double", num_params=1
)
kappa_1 = kappa_1_param.get_symbolic_repr()
kappa_2 = kappa_2_param.get_symbolic_repr()


# ----- DUMMY VARIABLES
# NOTE: in original formulation, rho and J are "dummy" variables that are effectively constants
# we can set them up as scalars/vec3's here to use, or since they're defined in the RHS function we'll do pure symbols
# rho_e = dendrosym.dtypes.scalar("rho_e" + idx_str)
# J = dendrosym.dtypes.vec3("rho_e" + idx_str)

rho_e = sym.Symbol("rho_e[pp]")
J = [sym.Symbol("J0[pp]"), sym.Symbol("J1[pp]"), sym.Symbol("J2[pp]")]

# ----- FETCH DERIVATIVES SPECIFICATION
#
# These are the derivative functions that will be used throughout the program
# setting them stores them inside the dendro package
# ===
# first derivative is the gradient, and first argument is direction
d_ = dendrosym.nr.set_first_derivative("grad")
# second derivative is second order gradient, and first 2
# arguments are direction
# d2s_ = dendrosym.nr.set_second_derivative('grad2')
d2_ = dendrosym.nr.set_second_derivative("grad2")
# advective derivate, first argument is direction
ad_ = dendrosym.nr.set_advective_derivative("grad")
# and then we set the kreiss oliger dissipation
kod_ = dendrosym.nr.set_kreiss_oliger_dissipation("kograd")


###############################################################
#  initialize
###############################################################

from sympy import LeviCivita

# declare functions


###############################################################
#  evolution equations
###############################################################
def em4_equations():
    B_rhs = [
        -sum(
            sum(LeviCivita(i, j, k) * d_(j, E[k]) for k in dendro.e_i)
            for j in dendro.e_i
        )
        + d_(i, Phi)
        for i in dendrosym.nr.e_i
    ]

    E_rhs = [
        sum(
            sum(LeviCivita(i, j, k) * d(j, B[k]) for k in dendro.e_i)
            for j in dendro.e_i
        )
        - 4.0 * PI * J[i]
        - d_(i, Psi)
        for i in dendrosym.nr.e_i
    ]

    Phi_rhs = sum(d_(i, B[i]) for i in dendrosym.nr.e_i) - kappa_2 * Phi

    Psi_rhs = (
        4 * PI * rho_e - sum(d_(i, E[i]) for i in dendrosym.nr.e_i) - kappa_1 * Psi
    )

    # list the variables in the same order as their RHS counterparts
    rhs_list = [B_rhs, E_rhs, Phi_rhs, Psi_rhs]
    var_list = [B, E, Phi, Psi]

    return rhs_list, var_list


dendroConfigs.set_rhs_equation_function("evolution", em4_equations)

# === TEMPORARY EXTRACTION CODE
# ==========================================
evolution_var_extraction = dendroConfigs.generate_variable_extraction(
    "evolution", use_const=True
)
evolution_var_rhs_extraction = dendroConfigs.generate_rhs_var_extraction(
    "evolution", zip_var_name="unzipVarsRHS"
)

print("// EVOLUTION VARIABLE EXTRACTION CODE")
print(evolution_var_extraction)
print(evolution_var_rhs_extraction)

evolution_parameters = dendroConfigs.gen_parameter_code("evolution")
print()
print("// PARAMETER EXTRACTION")
print(evolution_parameters)


# could replace derivative with stencil code
# dendroConfigs.replace_derivatives_with_stencil("evolution", 6)
# dendroConfigs.replace_derivatives_with_stencil("constraint", 6)

(
    intermediate_grad_str,
    deallocate_intermediate_grad_str,
) = dendroConfigs.generate_pre_necessary_derivatives(
    "evolution", dtype="double", include_byte_declaration=False
)

print()
# print(intermediate_grad_str)

(
    deriv_alloc,
    deriv_calc,
    deriv_dealloc,
) = dendroConfigs.generate_deriv_allocation_and_calc(
    "evolution", include_byte_declaration=False
)

with open("deriv_alloc.cpp", "w") as f:
    f.write(deriv_alloc)

with open("deriv_calc.cpp", "w") as f:
    f.write(deriv_calc)

with open("deriv_dealloc.cpp", "w") as f:
    f.write(deriv_dealloc)

evolution_rhs_code = dendroConfigs.generate_rhs_code("evolution")

with open("temporary_rhs_output.cpp", "w") as f:
    f.write(evolution_rhs_code)

with open("boundary_conds.cpp", "w") as f:
    f.write(dendroConfigs.generate_bcs_calculations("evolution"))


with open("koderivs.cpp", "w") as f:
    f.write(dendroConfigs.generate_ko_derivs("evolution"))

with open("kocalc.cpp", "w") as f:
    f.write(dendroConfigs.generate_ko_calculations("evolution"))
