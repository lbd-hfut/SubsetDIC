#include "types.hpp"
#include <cmath>

extern "C" {

bool interp_qbs_lut(const double* lut, int lut_h, int lut_w,
                    double x_tilda, double y_tilda, double& interp) {
    int x0 = (int)std::floor(x_tilda);
    int y0 = (int)std::floor(y_tilda);
    if (x0 < 0 || y0 < 0 || x0 >= lut_w || y0 >= lut_h) {
        interp = 0.0;
        return false;
    }

    double dx = x_tilda - (double)x0;
    double dy = y_tilda - (double)y0;

    double xpow[6] = {1.0, dx, dx * dx, 0.0, 0.0, 0.0};
    double ypow[6] = {1.0, dy, dy * dy, 0.0, 0.0, 0.0};
    for (int i = 3; i < 6; i++) {
        xpow[i] = xpow[i - 1] * dx;
        ypow[i] = ypow[i - 1] * dy;
    }

    const double* coeff = lut + ((y0 * lut_w + x0) * 36);
    interp = 0.0;
    for (int m = 0; m < 6; m++) {
        for (int n = 0; n < 6; n++) {
            interp += ypow[m] * coeff[m * 6 + n] * xpow[n];
        }
    }
    return true;
}

} // extern "C"
