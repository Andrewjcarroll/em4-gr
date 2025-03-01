
#include <Eigen/SparseCore>
#include <bitset>
#include <cstdint>
#include <iostream>
#include <random>
#include <toml.hpp>
#include <tuple>
#include <vector>

#include "compact_derivs.h"
#include "derivs.h"
#include "profiler.h"

#define RED "\e[1;31m"
#define BLU "\e[2;34m"
#define GRN "\e[0;32m"
#define YLW "\e[0;33m"
#define MAG "\e[0;35m"
#define CYN "\e[0;36m"
#define NRM "\e[0m"

typedef Eigen::SparseMatrix<double> SparseMat;
typedef Eigen::Triplet<double> Trip;

int main(int argc, char *argv[]) {
    std::cout << "hi" << std::endl;

    int n = 30;
    int rows = n;
    int cols = n;
    int m = n * n;

    // with eigen it's best to build a list of triplets and then convert it to a
    // sparse matrix

    std::vector<Trip> tripletList;

    // rough estimate of how many rows we're actually going to be using here...
    tripletList.reserve(rows * 3);

    for (size_t ii = 0; ii < rows; ii++) {
        if (ii != 0) {
            // then we do col ii - 1
            tripletList.push_back(Trip(ii, ii - 1, 0.5));
        }

        tripletList.push_back(Trip(ii, ii, 1.0));

        if (ii != rows - 1) {
            // do col ii + 1
            tripletList.push_back(Trip(ii, ii + 1, 0.5));
        }
    }

    // convert the triplet list to more
    SparseMat mat(rows, cols);
    mat.setFromTriplets(tripletList.begin(), tripletList.end());

    for (size_t k = 0; k < mat.outerSize(); ++k) {
        for (SparseMat::InnerIterator it(mat, k); it; ++it) {
            std::cout << "val, row, col, idx: " << it.value() << " " << it.row()
                      << " " << it.col() << " " << it.index() << std::endl;
        }
    }

    for (size_t k = 0; k < rows; k++) {
        for (size_t i = 0; i < cols; i++) {
            std::cout << mat.coeffRef(i, k) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    return 0;
}
