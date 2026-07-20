#include "types.hpp"
#include <cmath>
#include <algorithm>

extern "C" {

void cholesky(double* mat, bool& positivedef, int size_mat) {
    for (int i = 0; i < size_mat; i++) {
        if (i > 0) {
            for (int j = size_mat - 1; j >= i; j--) {
                double buf = 0.0;
                for (int k = 0; k < i; k++) {
                    buf += mat[j + k * size_mat] * mat[i + k * size_mat];
                }
                mat[j + i * size_mat] -= buf;
            }
        }
        if (mat[i + i * size_mat] > LAMBDA) {
            double diag_sqrt = std::sqrt(mat[i + i * size_mat]);
            for (int j = i; j < size_mat; j++) {
                mat[j + i * size_mat] /= diag_sqrt;
            }
        } else {
            positivedef = false;
            break;
        }
    }
}

void forwardsub(double* vec, const double* mat, int size_mat) {
    vec[0] /= mat[0];
    for (int i = 1; i < size_mat; i++) {
        double buf = 0.0;
        for (int j = 0; j < i; j++) {
            buf += mat[i + j * size_mat] * vec[j];
        }
        vec[i] = (vec[i] - buf) / mat[i + i * size_mat];
    }
}

void backwardsub(double* vec, const double* mat, int size_mat) {
    vec[size_mat - 1] /= mat[(size_mat - 1) + (size_mat - 1) * size_mat];
    for (int i = size_mat - 2; i > -1; i--) {
        double buf = 0.0;
        for (int j = i + 1; j < size_mat; j++) {
            buf += mat[j + i * size_mat] * vec[j];
        }
        vec[i] = (vec[i] - buf) / mat[i + i * size_mat];
    }
}

} // extern "C"
