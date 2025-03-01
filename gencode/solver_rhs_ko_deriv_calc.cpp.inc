/**
 * This file is a basic include
 *
 *
 */

/*[[[cog
import cog
import sys
import importlib.util
import dendrosym

# the following lines will import any module directly from
spec = importlib.util.spec_from_file_location("dendroconf", CONFIG_FILE_PATH)
dendroconf = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = dendroconf
spec.loader.exec_module(dendroconf)

cog.outl('//// KO GRADIENT CALCULATION')
cog.outl(dendroconf.dendroConfigs.generate_ko_derivs("evolution"))

]]]*/
//// KO GRADIENT CALCULATION

dendro_derivs::ko_deriv_x(grad_0_E0, E0, hx, sz, bflag);
dendro_derivs::ko_deriv_y(grad_1_E0, E0, hy, sz, bflag);
dendro_derivs::ko_deriv_z(grad_2_E0, E0, hz, sz, bflag);

dendro_derivs::ko_deriv_x(grad_0_E1, E1, hx, sz, bflag);
dendro_derivs::ko_deriv_y(grad_1_E1, E1, hy, sz, bflag);
dendro_derivs::ko_deriv_z(grad_2_E1, E1, hz, sz, bflag);

dendro_derivs::ko_deriv_x(grad_0_E2, E2, hx, sz, bflag);
dendro_derivs::ko_deriv_y(grad_1_E2, E2, hy, sz, bflag);
dendro_derivs::ko_deriv_z(grad_2_E2, E2, hz, sz, bflag);

dendro_derivs::ko_deriv_x(grad_0_B0, B0, hx, sz, bflag);
dendro_derivs::ko_deriv_y(grad_1_B0, B0, hy, sz, bflag);
dendro_derivs::ko_deriv_z(grad_2_B0, B0, hz, sz, bflag);

dendro_derivs::ko_deriv_x(grad_0_B1, B1, hx, sz, bflag);
dendro_derivs::ko_deriv_y(grad_1_B1, B1, hy, sz, bflag);
dendro_derivs::ko_deriv_z(grad_2_B1, B1, hz, sz, bflag);

dendro_derivs::ko_deriv_x(grad_0_B2, B2, hx, sz, bflag);
dendro_derivs::ko_deriv_y(grad_1_B2, B2, hy, sz, bflag);
dendro_derivs::ko_deriv_z(grad_2_B2, B2, hz, sz, bflag);

dendro_derivs::ko_deriv_x(grad_0_Phi, Phi, hx, sz, bflag);
dendro_derivs::ko_deriv_y(grad_1_Phi, Phi, hy, sz, bflag);
dendro_derivs::ko_deriv_z(grad_2_Phi, Phi, hz, sz, bflag);

dendro_derivs::ko_deriv_x(grad_0_Psi, Psi, hx, sz, bflag);
dendro_derivs::ko_deriv_y(grad_1_Psi, Psi, hy, sz, bflag);
dendro_derivs::ko_deriv_z(grad_2_Psi, Psi, hz, sz, bflag);

//[[[end]]]
