#pragma once

#include <stdint.h>

#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <vector>

#ifdef EM4_USE_XSMM_MAT_MUL
#include <libxsmm.h>
#endif

#include "dendro.h"

#define INDEX_3D(i, j, k) ((i) + nx * ((j) + ny * (k)))

#define INDEX_2D(i, j) ((i) + n * (j))

#define INDEX_N2D(i, j, n) ((i) + (n) * (j))

extern "C" {
/**
 * @brief LU decomposition of a general matrix.
 *
 * This function performs LU decomposition of a general matrix. See the LAPACK
 * documentation for more information.
 *
 * @param[in] n Number of rows of the matrix.
 * @param[in] m Number of columns of the matrix.
 * @param[in,out] P Matrix to be decomposed. On output, it contains the LU
 * decomposition.
 * @param[in] lda Leading dimension of the array P. Must be at least max(1, n).
 * @param[out] IPIV Array of pivot indices representing the permutation matrix.
 * @param[out] INFO INFO=0 indicates successful execution.
 */
void dgetrf_(int *n, int *m, double *P, int *lda, int *IPIV, int *INFO);

/**
 * @brief Generates the inverse of a matrix given its LU decomposition.
 *
 * This function generates the inverse of a matrix based on its LU
 * decomposition. See the LAPACK documentation for more information
 *
 * @param[in] N Order of the matrix.
 * @param[in,out] A Matrix containing the LU decomposition. On output, it
 * contains the inverse of the original matrix.
 * @param[in] lda Leading dimension of the array A. Must be at least max(1, N).
 * @param[in] IPIV Array of pivot indices representing the permutation matrix
 * from the LU decomposition.
 * @param[out] WORK Workspace array.
 * @param[in] lwork Size of the WORK array. lwork >= max(1, N).
 * @param[out] INFO INFO=0 indicates successful execution.
 */
void dgetri_(int *N, double *A, int *lda, int *IPIV, double *WORK, int *lwork,
             int *INFO);

/**
 * @brief Multiplies two matrices C = alpha*A*B + beta*C.
 *
 * This function performs matrix multiplication: C = alpha*A*B + beta*C. See the
 * LAPACK documentation for more information.
 *
 * @param[in] TA Indicates whether to transpose matrix A. 'N' for no transpose,
 * 'T' for transpose.
 * @param[in] TB Indicates whether to transpose matrix B. 'N' for no transpose,
 * 'T' for transpose.
 * @param[in] M Number of rows in matrices A and C.
 * @param[in] N Number of columns in matrices B and C.
 * @param[in] K Number of columns in matrix A and rows in matrix B.
 * @param[in] ALPHA Scalar multiplier for matrix A*B.
 * @param[in] A Matrix A with dimensions (LDA, K) if TA='N', (K, LDA) if TA='T'.
 * @param[in] LDA Leading dimension of array A. Must be at least max(1, (TA='N')
 * ? M : K).
 * @param[in] B Matrix B with dimensions (LDB, N) if TB='N', (N, LDB) if TB='T'.
 * @param[in] LDB Leading dimension of array B. Must be at least max(1, (TB='N')
 * ? K : N).
 * @param[in] BETA Scalar multiplier for matrix C.
 * @param[in,out] C Matrix C with dimensions (LDC, N).
 * @param[in] LDC Leading dimension of array C. Must be at least max(1, M).
 */
void dgemm_(char *TA, char *TB, int *M, int *N, int *K, double *ALPHA,
            double *A, int *LDA, double *B, int *LDB, double *BETA, double *C,
            int *LDC);

/**
 * @brief Generic matrix-vector multiplication.
 *
 * This function performs matrix-vector multiplication: y = alpha*A*x + beta*y.
 *
 * @param[in] trans Indicates whether to transpose matrix A. 'N' for no
 * transpose, 'T' for transpose.
 * @param[in] m Number of rows in matrix A.
 * @param[in] n Number of columns in matrix A.
 * @param[in] alpha Scalar multiplier for matrix-vector product.
 * @param[in] A Matrix A with dimensions (LDA, n) if trans='N', (m, LDA) if
 * trans='T'.
 * @param[in] lda Leading dimension of array A. Must be at least max(1,
 * (trans='N') ? m : n).
 * @param[in] x Vector x with at least (1+(n-1)*abs(incx)) elements if
 * trans='N', (1+(m-1)*abs(incx)) elements if trans='T'.
 * @param[in] incx Increment for the elements of vector x.
 * @param[in] beta Scalar multiplier for vector y.
 * @param[in,out] y Vector y with at least (1+(m-1)*abs(incy)) elements.
 * @param[in] incy Increment for the elements of vector y.
 */
void dgemv_(char *trans, int *m, int *n, double *alpha, double *A, int *lda,
            double *x, int *incx, double *beta, double *y, int *incy);
}

namespace dendro_cfd {

// enum DerType { CFD_P1_O4 = 0, CFD_P1_O6, CFD_Q1_O6_ETA1 };

enum DerType {
    // NO CFD Initialization
    CFD_NONE = -1,

    // the "main" compact finite difference types
    CFD_P1_O4 = 0,
    CFD_P1_O6,
    CFD_Q1_O6_ETA1,
    // isotropic finite difference types
    CFD_KIM_O4,
    CFD_HAMR_O4,
    CFD_JT_O6,

    // Explicit options using matrix mult
    EXPLCT_FD_O4,
    EXPLCT_FD_O6,
    EXPLCT_FD_O8,

    // Implicit 6th order derivatives as defined by BL
    CFD_BL_O4,
    CFD_BL_O6,
    CFD_BL_O8,

    // additional "helpers" that are mostly for internal/edge building
    CFD_DRCHLT_ORDER_4,
    CFD_DRCHLT_ORDER_6,
    CFD_P1_O4_CLOSE,
    CFD_P1_O6_CLOSE,
    CFD_P1_O4_L4_CLOSE,
    CFD_P1_O6_L6_CLOSE,
    CFD_Q1_O6,
    CFD_Q1_O6_CLOSE,
    CFD_DRCHLT_Q6,
    CFD_DRCHLT_Q6_L6,
    CFD_Q1_O6_ETA1_CLOSE,

};

// NOTE: BE SURE TO UPDATE THIS IF CHANGING ABOVE!
static const char *DER_TYPE_NAMES[] = {
    "CFD_NONE",
    "CFD_P1_O4",
    "CFD_P1_O6",
    "CFD_Q1_O6_ETA1",
    "CFD_KIM_O4",
    "CFD_HAMR_O4",
    "CFD_JT_O6",
    "EXPLCT_FD_O4",
    "EXPLCT_FD_O6",
    "EXPLCT_FD_O8",

    "CFD_DRCHLT_ORDER_4",
    "CFD_DRCHLT_ORDER_6",
    "CFD_P1_O4_CLOSE",
    "CFD_P1_O6_CLOSE",
    "CFD_P1_O4_L4_CLOSE",
    "CFD_P1_O6_L6_CLOSE",
    "CFD_Q1_O6",
    "CFD_Q1_O6_CLOSE",
    "CFD_DRCHLT_Q6",
    "CFD_DRCHLT_Q6_L6",
    "CFD_Q1_O6_ETA1_CLOSE",
};

enum DerType2nd {
    // NO CFD Initialization
    CFD2ND_NONE = -1,

    // the "main" compact finite difference types
    CFD2ND_P2_O4 = 0,
    CFD2ND_P2_O6,
    CFD2ND_Q2_O6_ETA1,
    // isotropic finite difference types
    CFD2ND_KIM_O4,   // FIX: KIM second orders aren't supported yet
    CFD2ND_HAMR_O4,  // FIX: HAMR second order isn't supported yet
    CFD2ND_JT_O6,    // FIX: JT second order isn't supported yet

    // explicit options using matrix mult
    EXPLCT2ND_FD_O4,
    EXPLCT2ND_FD_O6,
    EXPLCT2ND_FD_O8,

    // additional "helpers" that are mostly for internal/edge building
    CFD2ND_DRCHLT_ORDER_4,
    CFD2ND_DRCHLT_ORDER_6,
    CFD2ND_P2_O4_CLOSE,
    CFD2ND_P2_O6_CLOSE,
    CFD2ND_P2_O4_L4_CLOSE,
    CFD2ND_P2_O6_L6_CLOSE,
    CFD2ND_Q2_O6,
    CFD2ND_Q2_O6_CLOSE,
    CFD2ND_DRCHLT_Q6,
    CFD2ND_DRCHLT_Q6_L6,
    CFD2ND_Q2_O6_ETA1_CLOSE,

};

// NOTE: BE SURE TO UPDATE THIS IF CHANGING ABOVE!
static const char *DER_TYPE_2ND_NAMES[] = {
    "CFD2ND_NONE",     "CFD2ND_P2_O4",    "CFD2ND_P2_O6", "CFD2ND_Q2_O6_ETA1",
    "CFD2ND_KIM_O4",   "CFD2ND_HAMR_O4",  "CFD2ND_JT_O6", "EXPLCT2ND_FD_O4",
    "EXPLCT2ND_FD_O6", "EXPLCT2ND_FD_O8",
};

enum FilterType {
    // NO CFD Initialization
    FILT_NONE = -1,

    // standard filters...
    FILT_KO_DISS = 0,

    // isotropic finite difference types
    FILT_KIM_6,
    FILT_JT_6,
    FILT_JT_8,
    FILT_JT_10,

    // SBP Filter Option (derived KO Diss)
    FILT_SBP_FILTER,

    // explicit ko diss
    EXPLCT_KO,

};

// NOTE: BE SURE TO UPDATE THIS IF CHANGING ABOVE!
static const char *FILT_TYPE_NAMES[] = {
    "FILT_NONE", "FILT_KO_DISS", "FILT_KIM_6",      "FILT_JT_6",
    "FILT_JT_8", "FILT_JT_10",   "FILT_SBP_FILTER", "EXPLCT_KO",
};

// NOTE: these are going to be used as global parameters if they're not physical
enum BoundaryType {
    BLOCK_CFD_CLOSURE = 0,  // closure gives better results but 6th requires 4
                            // points, and 4th requires 3 points
    BLOCK_CFD_DIRICHLET,
    BLOCK_CFD_LOPSIDE_CLOSURE,
    BLOCK_PHYS_BOUNDARY
};

static const char *BOUNDARY_TYPE_NAMES[] = {
    "BLOCK_CFD_CLOSURE", "BLOCK_CFD_DIRICHLET", "BLOCK_CFD_LOPSIDE_CLOSURE",
    "BLOCK_PHYS_BOUNDARY"};

class CFDMethod {
   public:
    DerType name;
    uint32_t order;
    int32_t Ld;
    int32_t Rd;
    int32_t Lf;
    int32_t Rf;
    double alpha[16];
    double a[16];

    CFDMethod(DerType dertype) {
        switch (dertype) {
            case CFD_P1_O4:
                set_for_CFD_P1_O4();
                break;

            case CFD_P1_O6:
                set_for_CFD_P1_O6();
                break;

            case CFD_Q1_O6_ETA1:
                set_for_CFD_Q1_O6_ETA1();
                break;

            default:
                throw std::invalid_argument(
                    "Invalid CFD method type of " + std::to_string(dertype) +
                    " for initializing the CFDMethod Object");
                break;
        }
    }

    ~CFDMethod() {}

    void set_for_CFD_P1_O4() {
        name = CFD_P1_O4;
        order = 4;
        Ld = 1;
        Rd = 1;
        Lf = 1;
        Rf = 1;
        alpha[0] = 0.25;
        alpha[1] = 1.0;
        alpha[2] = 0.25;

        a[0] = -0.75;
        a[1] = 0.0;
        a[2] = 0.75;
    }

    void set_for_CFD_P1_O6() {
        name = CFD_P1_O6;
        order = 6;
        Ld = 1;
        Rd = 1;
        Lf = 2;
        Rf = 2;

        alpha[0] = 1.0 / 3.0;
        alpha[1] = 1.0;
        alpha[2] = 1.0 / 3.0;

        const double t1 = 1.0 / 36.0;
        a[0] = -t1;
        a[1] = -28.0 * t1;
        a[2] = 0.0;
        a[3] = 28.0 * t1;
        a[4] = t1;
    }

    void set_for_CFD_Q1_O6_ETA1() {
        name = CFD_Q1_O6_ETA1;
        order = 6;
        Ld = 1;
        Rd = 1;
        Lf = 3;
        Rf = 3;

        alpha[0] = 0.37987923;
        alpha[1] = 1.0;
        alpha[2] = 0.37987923;

        a[0] = 0.0023272948;
        a[1] = -0.052602255;
        a[2] = -0.78165660;
        a[3] = 0.0;
        a[4] = 0.78165660;
        a[5] = 0.052602255;
        a[6] = -0.0023272948;
    }
};

class CFDMethod2nd {
   public:
    DerType2nd name;
    uint32_t order;
    int32_t Ld;
    int32_t Rd;
    int32_t Lf;
    int32_t Rf;
    double alpha[16];
    double a[16];

    CFDMethod2nd(DerType2nd dertype) {
        switch (dertype) {
            case CFD2ND_P2_O4:
                set_for_CFD_P2_O4();
                break;

            case CFD2ND_P2_O6:
                set_for_CFD_P2_O6();
                break;

            case CFD2ND_Q2_O6_ETA1:
                set_for_CFD_Q2_O6_ETA1();
                break;

            default:
                throw std::invalid_argument(
                    "Invalid CFD 2nd order method type of " +
                    std::to_string(dertype) +
                    " for initializing the CFDMethod2nd Object");
                break;
        }
    }

    ~CFDMethod2nd() {}

    void set_for_CFD_P2_O4() {
        name = CFD2ND_P2_O4;
        order = 4;
        Ld = 1;
        Rd = 1;
        Lf = 1;
        Rf = 1;
        alpha[0] = 0.1;
        alpha[1] = 1.0;
        alpha[2] = 0.1;

        a[0] = 6.0 / 5.0;
        a[1] = -12.0 / 5.0;
        a[2] = 6.0 / 5.0;
    }

    void set_for_CFD_P2_O6() {
        name = CFD2ND_P2_O6;
        order = 6;
        Ld = 1;
        Rd = 1;
        Lf = 2;
        Rf = 2;

        alpha[0] = 2.0 / 11.0;
        alpha[1] = 1.0;
        alpha[2] = 2.0 / 11.0;

        const double t1 = 1.0 / 44.0;
        a[0] = 3.0 * t1;
        a[1] = 48.0 * t1;
        a[2] = -102.0 * t1;
        a[3] = 48.0 * t1;
        a[4] = 3.0 * t1;
    }

    void set_for_CFD_Q2_O6_ETA1() {
        name = CFD2ND_Q2_O6_ETA1;
        order = 6;
        Ld = 1;
        Rd = 1;
        Lf = 3;
        Rf = 3;

        alpha[0] = 0.24246603;
        alpha[1] = 1.0;
        alpha[2] = 0.24246603;

        a[0] = -0.0037062571;
        a[1] = 0.14095923;
        a[2] = 0.95445144;
        a[3] = -2.1834088;
        a[4] = 0.95445144;
        a[5] = 0.14095923;
        a[6] = -0.0037062571;
    }
};

/**
 * @brief Prints the elements of a square matrix.
 *
 * @param[in] m Pointer to the first element of the square matrix.
 * @param[in] n Size of ONE dimension of the matrix (i.e. n x n)
 */
void print_square_mat(double *m, const uint32_t n);

/**
 * @brief Prints the elements of a non-square matrix.
 *
 * @param[in] m Pointer to the first element of the square matrix.
 * @param[in] n_cols Size of the columns of the matrix
 * @param[in] n_rows Size of the rows of the matrix
 */
void print_nonsquare_mat(double *m, const uint32_t n_cols,
                         const uint32_t n_rows);

/**
 * @brief Determines what derivative type to use at edges.
 *
 * @param[in] derivtype The "main" derivative type.
 * @param[in] boundary The type of boundary that should be considered.
 */
DerType getDerTypeForEdges(const DerType derivtype,
                           const BoundaryType boundary);

/**
 * @brief Determines what derivative type to use at edges (2nd order).
 *
 * @param[in] derivtype The "main" second-order derivative type.
 * @param[in] boundary The type of boundary that should be considered.
 */
DerType2nd get2ndDerTypeForEdges(const DerType2nd derivtype,
                                 const BoundaryType boundary);

/**
 * @brief Builds matrices P and Q for a given derivative type.
 *
 * This function constructs matrices P and Q for a specified derivative type.
 * These are then used to generate the R matrix in another function.
 *
 * @param[out] P Pointer to the first element of matrix P.
 * @param[out] Q Pointer to the first element of matrix Q.
 * @param[in] padding Padding that should be used around the matrices.
 * @param[in] n Size of the square matrices (i.e. n x n).
 * @param[in] derivtype Type of derivative to construct the matrices for.
 * @param[in] is_left_edge Flag indicating if the operation is at the left edge.
 * @param[in] is_right_edge Flag indicating if the operation is at the right
 * edge.
 */
void buildPandQMatrices(
    double *P, double *Q, const uint32_t padding, const uint32_t n,
    const DerType derivtype, const bool is_left_edge = false,
    const bool is_right_edge = false,
    const BoundaryType boundary_type = BoundaryType::BLOCK_CFD_DIRICHLET);

/**
 * @brief Builds matrices P and Q for a given second-order derivative type.
 *
 * This function constructs matrices P and Q for a specified second-order
 * derivative type. These are then used to generate the R matrix in another
 * function.
 *
 * @param[out] P Pointer to the first element of matrix P.
 * @param[out] Q Pointer to the first element of matrix Q.
 * @param[in] padding Padding that should be used around the matrices.
 * @param[in] n Size of the square matrices (i.e. n x n).
 * @param[in] derivtype Type of derivative to construct the matrices for.
 * @param[in] is_left_edge Flag indicating if the operation is at the left edge.
 * @param[in] is_right_edge Flag indicating if the operation is at the right
 * edge.
 */
void buildPandQMatrices2ndOrder(
    double *P, double *Q, const uint32_t padding, const uint32_t n,
    const DerType2nd derivtype, const bool is_left_edge = false,
    const bool is_right_edge = false,
    const BoundaryType boundary_type = BoundaryType::BLOCK_CFD_DIRICHLET);

/**
 * @brief Builds matrices P and Q for a given filter type.
 *
 * This function constructs matrices P and Q for a specified filter type. These
 * are then used to generate the R matrix in another function.
 *
 * @param[out] P Pointer to the first element of matrix P.
 * @param[out] Q Pointer to the first element of matrix Q.
 * @param[in] padding Padding that should be used around the matrices.
 * @param[in] n Size of the square matrices (i.e. n x n).
 * @param[in] filtertype Type of filter to construct the matrices for.
 * @param[in] is_left_edge Flag indicating if the operation is at the left edge.
 * @param[in] is_right_edge Flag indicating if the operation is at the right
 * @param[in] kim_filt_kc Parameter for Kim filter to limit frequencies.
 * @param[in] kim_filt_eps Parameter for Kim filter used in computation.
 * edge.
 */
void buildPandQFilterMatrices(double *P, double *Q, const uint32_t padding,
                              const uint32_t n, const FilterType filtertype,
                              const double alpha, const bool bound_enable,
                              const bool is_left_edge = false,
                              const bool is_right_edge = false,
                              const double kim_filt_kc = 0.88,
                              const double kim_filt_eps = 0.25);

void buildMatrixLeft(double *P, double *Q, int *xib, const DerType dtype,
                     const int nghosts, const int n);

void buildMatrixRight(double *P, double *Q, int *xie, const DerType dtype,
                      const int nghosts, const int n);

void buildMatrixLeft2nd(double *P, double *Q, int *xib, const DerType2nd dtype,
                        const int nghosts, const int n);

void buildMatrixRight2nd(double *P, double *Q, int *xie, const DerType2nd dtype,
                         const int nghosts, const int n);

void calculateDerivMatrix(double *D, double *P, double *Q, const int n);

void setArrToZero(double *Mat, const int n);

/**
 * @brief Computes the matrix-matrix multiplication: C := alpha * op(A) * op(B)
 * + beta * C.
 *
 * This function is essentially just a wrapper function for dgemm_ that handles
 * setting up the values properly for that function.
 *
 * @param[out] C Pointer to the first element of the resulting matrix C.
 * @param[in] A Pointer to the first element of matrix A.
 * @param[in] B Pointer to the first element of matrix B.
 * @param[in] na Number of rows in matrix A.
 * @param[in] nb Number of columns in matrix B.
 */
void mulMM(double *C, double *A, double *B, int na, int nb);

/**
 * The order in which we compute the various compact derivatives.
 *
 * These values are used to help differentiate (in the source code) which
 * derivative or filter the iteration is currently on. This helps with
 * conditionals. An end user doesn't need to know what these values are.
 */
enum CompactDerivValueOrder {
    DERIV_NORM = 0,       ///< First-order derivative with no boundary handling
    DERIV_LEFT,           ///< First-order derivative with left boundary
    DERIV_RIGHT,          ///< First-order derivative with right boundary
    DERIV_LEFTRIGHT,      ///< First-order derivative with left-right boundary
    DERIV_2ND_NORM,       ///< Second-order derivative with no boundary handling
    DERIV_2ND_LEFT,       ///< Second-order derivative with no left boundary
    DERIV_2ND_RIGHT,      ///< Second-order derivative with no right boundary
    DERIV_2ND_LEFTRIGHT,  ///< Second-order derivative with left-right boundary
    FILT_NORM,            ///< The filter matrix with no boundary handling
    FILT_LEFT,            ///< The filter matrix with left boundary
    FILT_RIGHT,           ///< The filter matrix with right boundary
    FILT_LEFTRIGHT,       ///< The filter matrix with left-right boundary
    R_MAT_END             ///< Used to mark the end of the enum.
};

class CompactFiniteDiff {
   private:
// STORAGE VARIABLES USED FOR THE DIFFERENT DIMENSIONS
// Assume that the blocks are all the same size (to start with)

// Storage for the R matrix operator (combined P and Q matrices in CFD)
#ifdef SOLVER_ENABLE_MERGED_BLOCKS
    std::unordered_map<uint32_t, std::vector<double *>> m_R_storage;
#else
    double *m_RMatrices[CompactDerivValueOrder::R_MAT_END] = {};
#endif

    // Temporary storage for operations in progress
    double *m_u1d = nullptr;
    double *m_u2d = nullptr;
    // Additional temporary storage for operations in progress
    double *m_du1d = nullptr;
    double *m_du2d = nullptr;

    // pointers for our two workspaces, to potentially be set externally
    double *m_du3d_block1 = nullptr;
    double *m_du3d_block2 = nullptr;
    unsigned int m_max_blk_sz = 0;

    // TODO: make this a parameter!
    uint16_t m_largest_fusion = 10;
    // if (m n k)^(1/3) <= this value, then it's a small matrix mult
    double m_small_mat_threshold = 64.0;

    // to check for initialization (not used)
    bool m_initialized_matrices = false;

    // storing the derivative and filter types internally
    // could just be the parameter types
    DerType m_deriv_type = CFD_KIM_O4;
    DerType2nd m_second_deriv_type = CFD2ND_P2_O4;
    FilterType m_filter_type = FILT_NONE;
    unsigned int m_curr_dim_size = 0;
    unsigned int m_padding_size = 0;
    BoundaryType m_deriv_boundary_type = BoundaryType::BLOCK_CFD_DIRICHLET;

    double m_beta_filt = 0.0;

    // TODO: make a method that can set these!
    double m_filt_alpha = 0.0;
    double m_filt_bound_enable = false;
    double m_kim_filt_kc = 0.88 * M_PI;
    double m_kim_filt_eps = 0.25;

    typedef std::vector<std::tuple<uint32_t, uint32_t>> vec_tuple_int;
    vec_tuple_int m_matrix_size_pairs;
    std::vector<uint32_t> m_available_r_sizes;

#ifdef EM4_USE_XSMM_MAT_MUL
    typedef libxsmm_mmfunction<double> kernel_type;

#ifdef SOLVER_ENABLE_MERGED_BLOCKS

    std::map<uint32_t, kernel_type *> m_kernel_storage;
    std::map<uint32_t, kernel_type *> m_kernel_transpose_storage;

    std::map<uint32_t, kernel_type *> m_kernel_filt_storage;
    std::map<uint32_t, kernel_type *> m_kernel_filt_transpose_storage;

    kernel_type *m_kernel_x;
    kernel_type *m_kernel_y;
    kernel_type *m_kernel_z;

    kernel_type *m_kernel_x_filt;
    kernel_type *m_kernel_y_filt;
    kernel_type *m_kernel_z_filt;
#else
#endif
#endif

   public:
    CompactFiniteDiff(const unsigned int dim_size,
                      const unsigned int padding_size,
                      const DerType deriv_type = CFD_KIM_O4,
                      const DerType2nd second_deriv_type = CFD2ND_P2_O4,
                      const FilterType filter_type = FILT_NONE);
    ~CompactFiniteDiff();

    void change_dim_size(const unsigned int dim_size);

    void initialize_cfd_3dblock_workspace(const unsigned int max_blk_sz);
    void delete_cfd_3dblock_workspace();

    void initialize_cfd_storage();
    void initialize_all_cfd_matrices();
    void initialize_cfd_matrix(const uint32_t curr_size,
                               double **outputLocation);

    void add_cfd_matrix_to_storage(uint32_t size);

    void initialize_all_cfd_filters();
    void initialize_cfd_filter(const uint32_t curr_size,
                               double **outputLocation);
    void delete_cfd_matrices();

    void initialize_cfd_kernels();
    void delete_cfd_kernels();

    void calculate_sizes_that_work();

    void set_filter_type(FilterType filter_type) {
        m_filter_type = filter_type;
        if (m_filter_type == FilterType::FILT_KIM_6) {
            m_beta_filt = 1.0;
        } else {
            m_beta_filt = 0.0;
        }
    }

    void set_deriv_boundary_type(BoundaryType boundary_type) {
        m_deriv_boundary_type = boundary_type;
    }

    void set_kim_params(double kc, double eps) {
        m_kim_filt_kc = kc;
        m_kim_filt_eps = eps;
    }

    void set_deriv_type(const DerType deriv_type) {
        if (deriv_type != CFD_NONE && deriv_type != CFD_P1_O4 &&
            deriv_type != CFD_P1_O6 && deriv_type != CFD_Q1_O6_ETA1 &&
            deriv_type != CFD_KIM_O4 && deriv_type != CFD_HAMR_O4 &&
            deriv_type != CFD_JT_O6 && deriv_type != EXPLCT_FD_O4 &&
            deriv_type != EXPLCT_FD_O6 && deriv_type != EXPLCT_FD_O8 &&
            deriv_type != CFD_BL_O4 && deriv_type != CFD_BL_O6 &&
            deriv_type != CFD_BL_O8) {
            throw std::invalid_argument(
                "Couldn't change deriv type of CFD object, deriv type was not "
                "a valid "
                "'base' "
                "type: deriv_type = " +
                std::to_string(deriv_type));
        }
        m_deriv_type = deriv_type;
    }

    void set_second_deriv_type(const DerType2nd deriv_type) {
        if (deriv_type != CFD2ND_NONE && deriv_type != CFD2ND_P2_O4 &&
            deriv_type != CFD2ND_P2_O6 && deriv_type != CFD2ND_Q2_O6_ETA1 &&
            deriv_type != CFD2ND_KIM_O4 && deriv_type != CFD2ND_HAMR_O4 &&
            deriv_type != CFD2ND_JT_O6 && deriv_type != EXPLCT2ND_FD_O4 &&
            deriv_type != EXPLCT2ND_FD_O6 && deriv_type != EXPLCT2ND_FD_O8) {
            throw std::invalid_argument(
                "Couldn't change 2nd deriv type of CFD object, deriv type was "
                "not "
                "a valid "
                "'base' "
                "type: deriv_type = " +
                std::to_string(deriv_type));
        }
        m_second_deriv_type = deriv_type;
    }

    /**
     * Sets the padding size. NOTE however that this does *not* attempt to
     * regenerate the matrices, so be sure to call the initialization
     */
    void set_padding_size(const unsigned int padding_size) {
        m_padding_size = padding_size;
    }

    void clear_boundary_padding_nans(double *u, const unsigned int *sz,
                                     unsigned bflag);

    // the actual derivative computation side of things
    void cfd_x(double *const Dxu, const double *const u, const double dx,
               const unsigned int *sz, unsigned bflag);
    void cfd_y(double *const Dyu, const double *const u, const double dy,
               const unsigned int *sz, unsigned bflag);
    void cfd_z(double *const Dzu, const double *const u, const double dz,
               const unsigned int *sz, unsigned bflag);

    void cfd_xx(double *const Dxu, const double *const u, const double dx,
                const unsigned int *sz, unsigned bflag);
    void cfd_yy(double *const Dyu, const double *const u, const double dy,
                const unsigned int *sz, unsigned bflag);
    void cfd_zz(double *const Dzu, const double *const u, const double dz,
                const unsigned int *sz, unsigned bflag);

    // then the actual filters
    void filter_cfd_x(double *const u, double *const filtx_work,
                      const double dx, const unsigned int *sz, unsigned bflag);
    void filter_cfd_y(double *const u, double *const filty_work,
                      const double dy, const unsigned int *sz, unsigned bflag);
    void filter_cfd_z(double *const u, double *const filtz_work,
                      const double dz, const unsigned int *sz, unsigned bflag);
};

extern CompactFiniteDiff cfd;

extern unsigned int blderiv_matnum;

void set_bl_matnum(unsigned int input);
unsigned int get_bl_matnum();

/**
 * Initialization of various Compact Methods
 *
 * From this point on various compact finite methods can be calculated and
 * derived
 */

/**
 * Initialization of the P and Q matrices for Kim's 4th Order Compact
 * Derivatives
 *
 * P and Q are assumed to **already by zeroed out**.
 *
 * These derivative coefficients come from Tables I and II of :
 *
 * Jae Wook Kim, "Quasi-disjoint pentadiagonal matrix systems for
 * the parallelization of compact finite-difference schemes and
 * filters," Journal of Computational Physics 241 (2013) 168–194.
 */
void initializeKim4PQ(double *P, double *Q, int n);

/**
 * Initialization of the P and Q matrices for Kim's 6th Order Compact Filter
 *
 * P and Q are assumed to **already by zeroed out**.
 *
 * These filter coefficients come from Tables III and IV of :
 *
 * Jae Wook Kim, "Quasi-disjoint pentadiagonal matrix systems for
 * the parallelization of compact finite-difference schemes and
 * filters," Journal of Computational Physics 241 (2013) 168–194.
 */

// calculates some parts of the kim coefficients for us
void kim_filter_cal_coeff(double *c, double kc);

void initializeKim6FilterPQ(double *P, double *Q, int n, double kc = 0.88,
                            double eps = 0.25);

void initializeJTFilterT6PQ(double *P, double *Q, int n, int padding,
                            double alpha, bool fbound,
                            bool is_left_edge = false,
                            bool is_right_edge = false);

void initializeJTFilterT8PQ(double *P, double *Q, int n, int padding,
                            double alpha, bool fbound,
                            bool is_left_edge = false,
                            bool is_right_edge = false);

void initializeJTFilterT10PQ(double *P, double *Q, int n, int padding,
                             double alpha, bool fbound,
                             bool is_left_edge = false,
                             bool is_right_edge = false);
// KO explicit filters

void buildKOExplicitFilter(double *R, const unsigned int n,
                           const unsigned int padding, const unsigned int order,
                           bool is_left_edge, bool is_right_edge);

void buildKOExplicit6thOrder(double *R, const unsigned int n,
                             const unsigned int padding, bool is_left_edge,
                             bool is_right_edge);
void buildKOExplicit8thOrder(double *R, const unsigned int n,
                             const unsigned int padding, bool is_left_edge,
                             bool is_right_edge);

void buildDerivExplicitRMatrix(double *R, const unsigned int padding,
                               const unsigned int n, const DerType deriv_type,
                               const bool is_left_edge,
                               const bool is_right_edge);

void build2ndDerivExplicitRMatrix(double *R, const unsigned int padding,
                                  const unsigned int n,
                                  const DerType2nd deriv_type,
                                  const bool is_left_edge,
                                  const bool is_right_edge);

// explicit deriv operators
void buildDerivExplicit4thOrder(double *R, const unsigned int n,
                                bool is_left_edge, bool is_right_edge);

void buildDerivExplicit6thOrder(double *R, const unsigned int n,
                                bool is_left_edge, bool is_right_edge);

void buildDerivExplicit8thOrder(double *R, const unsigned int n,
                                bool is_left_edge, bool is_right_edge);

void build2ndDerivExplicit4thOrder(double *R, const unsigned int n,
                                   bool is_left_edge, bool is_right_edge);

void build2ndDerivExplicit6thOrder(double *R, const unsigned int n,
                                   bool is_left_edge, bool is_right_edge);

void build2ndDerivExplicit8thOrder(double *R, const unsigned int n,
                                   bool is_left_edge, bool is_right_edge);

class CFDNotImplemented : public std::exception {
   private:
    std::string message_;

   public:
    explicit CFDNotImplemented(const std::string &msg) : message_(msg) {}
    const char *what() { return message_.c_str(); }
};

}  // namespace dendro_cfd

/**
 * HAMR Derivatives and such Initialization
 */

/**
 * Initializes the "P" Matrix of the HAMR Derivatives
 *
 * @param P Pointer to the output "P" matrix
 * @param n The number of rows/cols of the square matrix
 */
void HAMRDeriv4_dP(double *P, int n);

/**
 * Initializes the "Q" Matrix of the HAMR Derivatives
 *
 * @param Q Pointer to the output "Q" matrix
 * @param n The number of rows/cols of the square matrix
 */
void HAMRDeriv4_dQ(double *Q, int n);

/**
 * Initializes the "R" Matrix of the HAMR Derivatives
 *
 * This is a combination function that can automatically call
 * dP and dQ to just give the R matrix.
 *
 * @param R Pointer to the output "R" matrix
 * @param n The number of rows/cols of the square matrix
 */
bool initHAMRDeriv4(double *R, const unsigned int n);

/**
 * JTP 6 Derivatives Initialization
 */

/**
 * Initializes the "P" Matrix of the JTP Derivatives
 *
 * @param P Pointer to the output "P" matrix
 * @param n The number of rows/cols of the square matrix
 */
void JTPDeriv6_dP(double *P, int n);

/**
 * Initializes the "Q" Matrix of the JTP Derivatives
 *
 * @param Q Pointer to the output "Q" matrix
 * @param n The number of rows/cols of the square matrix
 */
void JTPDeriv6_dQ(double *Q, int n);

/**
 * Initializes the "R" Matrix of the JTP Derivatives
 *
 * This is a combination function that can automatically call
 * dP and dQ to just give the R matrix.
 *
 * @param R Pointer to the output "R" matrix
 * @param n The number of rows/cols of the square matrix
 */
bool initJTPDeriv6(double *R, const unsigned int n);

/**
 * Kim Derivatives Initialization
 */

/**
 * Initializes the "P" Matrix of the Kim Derivatives
 *
 * NOTE: this method is depreciated in favor of initializeKim4PQ
 *
 * @param P Pointer to the output "P" matrix
 * @param n The number of rows/cols of the square matrix
 */
void KimDeriv4_dP(double *P, int n);

/**
 * Initializes the "Q" Matrix of the Kim Derivatives
 *
 * NOTE: this method is depreciated in favor of initializeKim4PQ
 *
 * @param Q Pointer to the output "Q" matrix
 * @param n The number of rows/cols of the square matrix
 */
void KimDeriv4_dQ(double *Q, int n);

/**
 * Initializes the "R" Matrix of the Kim Derivatives
 *
 * NOTE: this method is depreciated in favor of initializeKim4PQ
 *
 * @param R Pointer to the output "R" matrix
 * @param n The number of rows/cols of the square matrix
 */
bool initKimDeriv4(double *R, const unsigned int n);

/**
 * Initializes the "RF" Matrix of the Kim Filter
 *
 * NOTE: this method is depreciated in favor of initializeKim6FilterPQ
 *
 * @param RF Pointer to the output "RF" matrix
 * @param n The number of rows/cols of the square matrix
 */
bool initKim_Filter_Deriv4(double *RF, const unsigned int n);

/**
 * Initializes the coefficients for SBP Dissipation for order 2-1
 *
 * @param a A 16x16 matrix that stores the first set or coefficients
 * @param q A vector that stores the second set of coefficients
 */
void sbp_diss_2_1_coeffs(double a[16][16], double q[2]);

/**
 * Initializes the coefficients for SBP Dissipation for order 4-2
 *
 * @param a A 16x16 matrix that stores the first set or coefficients
 * @param q A vector that stores the second set of coefficients
 */
void sbp_diss_4_2_coeffs(double a[16][16], double q[3]);

/**
 * Initializes the coefficients for SBP Dissipation for order 6-3
 *
 * @param a A 16x16 matrix that stores the first set or coefficients
 * @param q A vector that stores the second set of coefficients
 */
void sbp_diss_6_3_coeffs(double a[16][16], double q[4]);

/**
 * Initializes the coefficients for SBP Dissipation for order 8-4
 *
 * @param a A 16x16 matrix that stores the first set or coefficients
 * @param q A vector that stores the second set of coefficients
 */
void sbp_diss_8_4_coeffs(double a[16][16], double q[5]);

void sbp_init_filter(double *A, unsigned int order, unsigned int n);

void get_bl_6th_order_values(uint32_t matrix_number, double alpha[4][6],
                             double beta[4][3]);

void BLDeriv_6OrderPQ(double *P, double *Q, int n, uint32_t scheme_number);
