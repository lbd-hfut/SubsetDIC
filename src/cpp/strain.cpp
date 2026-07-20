#include "types.hpp"
#include <algorithm>
#include <cstring>

extern "C" {

void cholesky(double* mat, bool& positivedef, int size_mat);
void forwardsub(double* vec, const double* mat, int size_mat);
void backwardsub(double* vec, const double* mat, int size_mat);
void get_cirroi(Cirroi& cirroi, const Region& region, int x, int y, int radius, bool subsettrunc);

int compute_strain(
    const double* plot_u,
    const double* plot_v,
    const Region* regions,
    int num_regions,
    const uint8_t* valid_mask,
    int grid_h, int grid_w,
    int radius_strain,
    int space,
    int subsettrunc,
    double* out_dudx,
    double* out_dudy,
    double* out_dvdx,
    double* out_dvdy,
    uint8_t* out_valid)
{
    int total = grid_h * grid_w;

    std::memcpy(out_valid, valid_mask, total * sizeof(uint8_t));

    std::fill(out_dudx, out_dudx + total, 0.0);
    std::fill(out_dudy, out_dudy + total, 0.0);
    std::fill(out_dvdx, out_dvdx + total, 0.0);
    std::fill(out_dvdy, out_dvdy + total, 0.0);

    double mat_LS[9];
    double u_vec[3];
    double v_vec[3];

    Cirroi cirroi;

    for (int r = 0; r < num_regions; r++) {
        const Region& region = regions[r];

        for (int j = 0; j < region.height_nodelist; j++) {
            int x = j + region.leftbound;
            if (x < 0 || x >= grid_w) continue;

            for (int k = 0; k < region.noderange[j]; k += 2) {
                for (int y = region.nodelist[j + k * region.height_nodelist];
                     y <= region.nodelist[j + (k + 1) * region.height_nodelist];
                     y++) {

                    if (y < 0 || y >= grid_h) continue;
                    int idx = y * grid_w + x;
                    if (!valid_mask[idx]) continue;

                    get_cirroi(cirroi, region, x, y, radius_strain, subsettrunc != 0);

                    std::fill(mat_LS, mat_LS + 9, 0.0);
                    std::fill(u_vec, u_vec + 3, 0.0);
                    std::fill(v_vec, v_vec + 3, 0.0);
                    int count = 0;

                    for (int m = 0; m < cirroi.region.height_nodelist; m++) {
                        int cx = m + x - cirroi.radius;
                        if (cx < 0 || cx >= grid_w) continue;

                        for (int n = 0; n < cirroi.region.noderange[m]; n += 2) {
                            for (int p = cirroi.region.nodelist[m + n * cirroi.region.height_nodelist];
                                 p <= cirroi.region.nodelist[m + (n + 1) * cirroi.region.height_nodelist];
                                 p++) {
                                int cy = p;
                                if (cy < 0 || cy >= grid_h) continue;

                                int cir_idx = cy * grid_w + cx;
                                if (!valid_mask[cir_idx]) continue;

                                double x_rel = (double)m - (double)cirroi.radius;
                                double y_rel = (double)p - (double)y;
                                double u_val = plot_u[cir_idx];
                                double v_val = plot_v[cir_idx];

                                mat_LS[0] += x_rel * x_rel;
                                mat_LS[3] += x_rel * y_rel;
                                mat_LS[4] += y_rel * y_rel;
                                mat_LS[6] += x_rel;
                                mat_LS[7] += y_rel;

                                u_vec[0] += x_rel * u_val;
                                u_vec[1] += y_rel * u_val;
                                u_vec[2] += u_val;

                                v_vec[0] += x_rel * v_val;
                                v_vec[1] += y_rel * v_val;
                                v_vec[2] += v_val;

                                count++;
                            }
                        }
                    }

                    if (count < 3) {
                        out_valid[idx] = 0;
                        continue;
                    }

                    mat_LS[1] = mat_LS[3];
                    mat_LS[2] = mat_LS[6];
                    mat_LS[5] = mat_LS[7];
                    mat_LS[8] = (double)count;

                    bool positivedef = true;
                    cholesky(mat_LS, positivedef, 3);
                    if (!positivedef) {
                        out_valid[idx] = 0;
                        continue;
                    }

                    forwardsub(u_vec, mat_LS, 3);
                    forwardsub(v_vec, mat_LS, 3);
                    backwardsub(u_vec, mat_LS, 3);
                    backwardsub(v_vec, mat_LS, 3);

                    if (space > 0) {
                        double scale = (double)space;
                        u_vec[0] *= scale;
                        u_vec[1] *= scale;
                        v_vec[0] *= scale;
                        v_vec[1] *= scale;
                    }

                    out_dudx[idx] = u_vec[0];
                    out_dudy[idx] = u_vec[1];
                    out_dvdx[idx] = v_vec[0];
                    out_dvdy[idx] = v_vec[1];
                }
            }
        }
    }

    return 0;
}

} // extern "C"
