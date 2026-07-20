#include "types.hpp"
#include <cmath>
#include <cstring>

// Simple bilinear interpolation for practical use.
// The LUT is a direct copy of the image (no B-spline precomputation).

extern "C" {

bool interp_qbs_lut(const double* lut, int lut_h, int lut_w,
                    double x_tilda, double y_tilda, double& interp) {
    int x0 = (int)std::floor(x_tilda);
    int y0 = (int)std::floor(y_tilda);
    if (x0 < 0 || y0 < 0 || x0 + 1 >= lut_w || y0 + 1 >= lut_h) {
        interp = 0.0;
        return false;
    }
    double dx = x_tilda - (double)x0;
    double dy = y_tilda - (double)y0;
    double v00 = lut[y0     + x0     * lut_h];
    double v10 = lut[y0     + (x0+1) * lut_h];
    double v01 = lut[y0 + 1 + x0     * lut_h];
    double v11 = lut[y0 + 1 + (x0+1) * lut_h];
    interp = (1.0-dx)*(1.0-dy)*v00 + dx*(1.0-dy)*v10 + (1.0-dx)*dy*v01 + dx*dy*v11;
    return true;
}

} // extern "C"
