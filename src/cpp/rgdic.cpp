#include "types.hpp"
#include <queue>
#include <vector>
#include <cmath>
#include <cstring>

extern "C" {

bool interp_qbs_lut(const double* lut, int lut_h, int lut_w,
                    double x_tilda, double y_tilda, double& interp);
void cholesky(double* mat, bool& positivedef, int size_mat);
void forwardsub(double* vec, const double* mat, int size_mat);
void backwardsub(double* vec, const double* mat, int size_mat);
void get_cirroi(Cirroi& cirroi, const Region& region, int x, int y, int radius, bool subsettrunc);
bool within_region(const Region& region, int x, int y);
int calc_seed_point(
    const double* ref_img, int ref_h, int ref_w,
    const double* ref_grad_x, const double* ref_grad_y,
    int grad_h, int grad_w,
    int border_bcoef,
    const double* cur_lut, int cur_lut_h, int cur_lut_w,
    const double* cur_img, int cur_h, int cur_w,
    const Region* regions, int num_regions,
    int seed_x, int seed_y,
    int radius, double cutoff_diffnorm, int cutoff_iteration, int subsettrunc,
    double* out_u, double* out_v, double* out_dudx, double* out_dudy,
    double* out_dvdx, double* out_dvdy,
    double* out_corrcoef, double* out_diffnorm, int* out_iterations);

struct QueueItem {
    double corrcoef;
    int grid_x, grid_y;
    double u, v;
    double du_dx, du_dy, dv_dx, dv_dy;
    bool operator<(const QueueItem& other) const {
        return corrcoef > other.corrcoef;
    }
};

int rgdic_analysis(
    const double* ref_img, int ref_h, int ref_w,
    const double* ref_grad_x, const double* ref_grad_y,
    int grad_h, int grad_w,
    int border_bcoef,
    const double* cur_lut, int cur_lut_h, int cur_lut_w,
    const double* cur_img, int cur_h, int cur_w,
    const Region* regions, int num_regions,
    const double* seed_params, const int* seed_xs, const int* seed_ys, int num_seeds,
    int radius, int spacing, double cutoff_diffnorm, int cutoff_iteration, int subsettrunc,
    int direct_seed_grid,
    double* out_u, double* out_v, double* out_corrcoef, uint8_t* out_valid,
    int out_h, int out_w, int* out_points_computed
) {
    int step = spacing < 1 ? 1 : spacing;
    const double cutoff_corrcoef = 0.1;
    const double cutoff_disp = 1.0;

    std::memset(out_u, 0, out_h * out_w * sizeof(double));
    std::memset(out_v, 0, out_h * out_w * sizeof(double));
    std::memset(out_corrcoef, 0, out_h * out_w * sizeof(double));
    std::memset(out_valid, 0, out_h * out_w * sizeof(uint8_t));

    std::vector<uint8_t> calculated(out_h * out_w, 0);

    int num_computed = 0;
    int diam = 2 * radius + 1;

    if (direct_seed_grid != 0) {
        for (int gy = 0; gy < out_h; gy++) {
            int y = gy * step;
            if (y < radius || y >= ref_h - radius)
                continue;
            for (int gx = 0; gx < out_w; gx++) {
                int x = gx * step;
                if (x < radius || x >= ref_w - radius)
                    continue;

                bool in_region = false;
                for (int r = 0; r < num_regions; r++) {
                    if (within_region(regions[r], x, y)) {
                        in_region = true;
                        break;
                    }
                }
                if (!in_region)
                    continue;

                double su = 0.0, sv = 0.0, sdudx = 0.0, sdudy = 0.0;
                double sdvdx = 0.0, sdvdy = 0.0, scorr = 0.0, sdiff = 0.0;
                int siter = 0;
                int sret = calc_seed_point(
                    ref_img, ref_h, ref_w,
                    ref_grad_x, ref_grad_y, grad_h, grad_w,
                    border_bcoef,
                    cur_lut, cur_lut_h, cur_lut_w,
                    cur_img, cur_h, cur_w,
                    regions, num_regions,
                    x, y, radius, cutoff_diffnorm, cutoff_iteration, subsettrunc,
                    &su, &sv, &sdudx, &sdudy, &sdvdx, &sdvdy,
                    &scorr, &sdiff, &siter);

                if (sret == 1 &&
                    scorr < 0.5 &&
                    sdiff <= 0.1 &&
                    std::fabs(su) <= radius + 1.0 &&
                    std::fabs(sv) <= radius + 1.0 &&
                    std::isfinite(su) && std::isfinite(sv) &&
                    std::isfinite(scorr)) {
                    int idx = gy * out_w + gx;
                    out_valid[idx] = 1;
                    out_u[idx] = su;
                    out_v[idx] = sv;
                    out_corrcoef[idx] = scorr;
                    num_computed++;
                }
            }
        }
        *out_points_computed = num_computed;
        return 1;
    }

    std::vector<double> g_buffer(diam * diam, 0.0);
    std::vector<double> df_dp_buffer(diam * diam * 6, 0.0);
    std::vector<double> gradient(6, 0.0);
    std::vector<double> hessian(36, 0.0);

    for (int s = 0; s < num_seeds; s++) {
        const double* sp = &seed_params[s * 6];
        int seed_x = seed_xs[s];
        int seed_y = seed_ys[s];

        int region_idx = -1;
        for (int r = 0; r < num_regions; r++) {
            if (within_region(regions[r], seed_x, seed_y)) {
                region_idx = r;
                break;
            }
        }
        if (region_idx < 0)
            continue;
        const Region& region = regions[region_idx];

        int gx_seed = seed_x / step;
        int gy_seed = seed_y / step;
        if (gx_seed < 0 || gx_seed >= out_w || gy_seed < 0 || gy_seed >= out_h)
            continue;

        std::priority_queue<QueueItem> queue;

        int sidx = gy_seed * out_w + gx_seed;
        calculated[sidx] = 1;
        out_valid[sidx] = 1;
        out_u[sidx] = sp[0];
        out_v[sidx] = sp[1];
        out_corrcoef[sidx] = 0.0;

        QueueItem seed_item;
        seed_item.corrcoef = 0.0;
        seed_item.grid_x = gx_seed;
        seed_item.grid_y = gy_seed;
        seed_item.u = sp[0];
        seed_item.v = sp[1];
        seed_item.du_dx = sp[2];
        seed_item.du_dy = sp[3];
        seed_item.dv_dx = sp[4];
        seed_item.dv_dy = sp[5];
        queue.push(seed_item);
        num_computed++;

        while (!queue.empty()) {
            QueueItem item = queue.top();
            queue.pop();

            int idx = item.grid_y * out_w + item.grid_x;
            out_u[idx] = item.u;
            out_v[idx] = item.v;
            out_corrcoef[idx] = item.corrcoef;

            int pop_x = item.grid_x * step;
            int pop_y = item.grid_y * step;

            int neighbors[4][2] = {
                {pop_x, pop_y - step},
                {pop_x + step, pop_y},
                {pop_x, pop_y + step},
                {pop_x - step, pop_y}
            };

            for (int n = 0; n < 4; n++) {
                int nx = neighbors[n][0];
                int ny = neighbors[n][1];
                if (nx < 0 || ny < 0 || nx >= ref_w || ny >= ref_h)
                    continue;

                int gnx = nx / step;
                int gny = ny / step;
                if (gnx < 0 || gny < 0 || gnx >= out_w || gny >= out_h)
                    continue;

                int nidx = gny * out_w + gnx;
                if (calculated[nidx])
                    continue;

                if (!within_region(region, nx, ny)) {
                    calculated[nidx] = 1;
                    continue;
                }

                double init_u = item.u + item.du_dx * (nx - pop_x) + item.du_dy * (ny - pop_y);
                double init_v = item.v + item.dv_dx * (nx - pop_x) + item.dv_dy * (ny - pop_y);

                Cirroi cirroi;
                get_cirroi(cirroi, region, nx, ny, radius, subsettrunc != 0);
                if (cirroi.region.totalpoints == 0) {
                    calculated[nidx] = 1;
                    continue;
                }

                int nr_h = cirroi.region.height_nodelist;
                int gbuf_size = nr_h * diam;
                if ((int)g_buffer.size() < gbuf_size)
                    g_buffer.resize(gbuf_size, 0.0);
                int dfdp_size = gbuf_size * 6;
                if ((int)df_dp_buffer.size() < dfdp_size)
                    df_dp_buffer.resize(dfdp_size, 0.0);

                double fm = 0.0;
                for (int ci = 0; ci < nr_h; ci++) {
                    for (int cj = 0; cj < cirroi.region.noderange[ci]; cj += 2) {
                        int top = cirroi.region.nodelist[ci + cj * nr_h];
                        int bot = cirroi.region.nodelist[ci + (cj + 1) * nr_h];
                        for (int ck = top; ck <= bot; ck++) {
                            fm += ref_img[ck + (ci + cirroi.x - cirroi.radius) * ref_h];
                        }
                    }
                }
                fm /= (double)cirroi.region.totalpoints;

                double deltaf_inv = 0.0;
                for (int ci = 0; ci < nr_h; ci++) {
                    for (int cj = 0; cj < cirroi.region.noderange[ci]; cj += 2) {
                        int top = cirroi.region.nodelist[ci + cj * nr_h];
                        int bot = cirroi.region.nodelist[ci + (cj + 1) * nr_h];
                        for (int ck = top; ck <= bot; ck++) {
                            double diff = ref_img[ck + (ci + cirroi.x - cirroi.radius) * ref_h] - fm;
                            deltaf_inv += diff * diff;
                        }
                    }
                }
                deltaf_inv = std::sqrt(deltaf_inv);
                if (deltaf_inv <= LAMBDA) {
                    calculated[nidx] = 1;
                    continue;
                }
                deltaf_inv = 1.0 / deltaf_inv;

                for (int ci = 0; ci < nr_h; ci++) {
                    int px = ci + cirroi.x - cirroi.radius;
                    for (int cj = 0; cj < cirroi.region.noderange[ci]; cj += 2) {
                        int top = cirroi.region.nodelist[ci + cj * nr_h];
                        int bot = cirroi.region.nodelist[ci + (cj + 1) * nr_h];
                        for (int ck = top; ck <= bot; ck++) {
                            int py = ck;
                            double ldx = (double)(ci - cirroi.radius);
                            double ldy = (double)(ck - cirroi.y);
                            int gidx = (py + (border_bcoef > 0 ? border_bcoef - 2 : 0)) + (px + (border_bcoef > 0 ? border_bcoef - 2 : 0)) * grad_h;
                            int lind_df = (ck - cirroi.y + cirroi.radius) * 6 + ci * (nr_h * 6);

                            double dfdx = ref_grad_x[gidx];
                            double dfdy = ref_grad_y[gidx];

                            df_dp_buffer[lind_df]     = dfdx;
                            df_dp_buffer[lind_df + 1] = dfdy;
                            df_dp_buffer[lind_df + 2] = dfdx * ldx;
                            df_dp_buffer[lind_df + 3] = dfdx * ldy;
                            df_dp_buffer[lind_df + 4] = dfdy * ldx;
                            df_dp_buffer[lind_df + 5] = dfdy * ldy;
                        }
                    }
                }

                std::fill(hessian.begin(), hessian.end(), 0.0);
                for (int ci = 0; ci < nr_h; ci++) {
                    for (int cj = 0; cj < cirroi.region.noderange[ci]; cj += 2) {
                        int top = cirroi.region.nodelist[ci + cj * nr_h];
                        int bot = cirroi.region.nodelist[ci + (cj + 1) * nr_h];
                        for (int ck = top; ck <= bot; ck++) {
                            int lind_df = (ck - cirroi.y + cirroi.radius) * 6 + ci * (nr_h * 6);
                            double* d = &df_dp_buffer[lind_df];
                            hessian[0]  += d[0] * d[0];
                            hessian[1]  += d[0] * d[1];
                            hessian[2]  += d[0] * d[2];
                            hessian[3]  += d[0] * d[3];
                            hessian[4]  += d[0] * d[4];
                            hessian[5]  += d[0] * d[5];
                            hessian[7]  += d[1] * d[1];
                            hessian[8]  += d[1] * d[2];
                            hessian[9]  += d[1] * d[3];
                            hessian[10] += d[1] * d[4];
                            hessian[11] += d[1] * d[5];
                            hessian[14] += d[2] * d[2];
                            hessian[15] += d[2] * d[3];
                            hessian[16] += d[2] * d[4];
                            hessian[17] += d[2] * d[5];
                            hessian[21] += d[3] * d[3];
                            hessian[22] += d[3] * d[4];
                            hessian[23] += d[3] * d[5];
                            hessian[28] += d[4] * d[4];
                            hessian[29] += d[4] * d[5];
                            hessian[35] += d[5] * d[5];
                        }
                    }
                }

                double mult = 2.0 * deltaf_inv * deltaf_inv;
                for (int hi = 0; hi < 6; hi++) {
                    for (int hj = hi; hj < 6; hj++) {
                        hessian[hj + hi * 6] *= mult;
                    }
                }

                hessian[6]  = hessian[1];
                hessian[12] = hessian[2];  hessian[13] = hessian[8];
                hessian[18] = hessian[3];  hessian[19] = hessian[9];  hessian[20] = hessian[15];
                hessian[24] = hessian[4];  hessian[25] = hessian[10]; hessian[26] = hessian[16]; hessian[27] = hessian[22];
                hessian[30] = hessian[5];  hessian[31] = hessian[11]; hessian[32] = hessian[17]; hessian[33] = hessian[23]; hessian[34] = hessian[29];

                bool positivedef = true;
                cholesky(hessian.data(), positivedef, 6);
                if (!positivedef) {
                    calculated[nidx] = 1;
                    continue;
                }

                double defvec[6] = {init_u, init_v, item.du_dx, item.du_dy, item.dv_dx, item.dv_dy};
                double corr_result = 0.0;
                bool converged = false;
                double diffnorm = 0.0;

                for (int iter = 0; iter <= cutoff_iteration; iter++) {
                    double gm = 0.0;
                    for (int ci = 0; ci < nr_h; ci++) {
                        int px = ci + cirroi.x - cirroi.radius;
                        for (int cj = 0; cj < cirroi.region.noderange[ci]; cj += 2) {
                            int top = cirroi.region.nodelist[ci + cj * nr_h];
                            int bot = cirroi.region.nodelist[ci + (cj + 1) * nr_h];
                            for (int ck = top; ck <= bot; ck++) {
                                int py = ck;
                                double ldx = (double)(ci - cirroi.radius);
                                double ldy = (double)(ck - cirroi.y);

                                double lut_offset = (border_bcoef > 0) ? (double)(border_bcoef - 2) : 0.0;
                                double x_tilda_bcoef = (double)px + lut_offset
                                    + defvec[0] + defvec[2] * ldx + defvec[3] * ldy;
                                double y_tilda_bcoef = (double)py + lut_offset
                                    + defvec[1] + defvec[4] * ldx + defvec[5] * ldy;

                                int lind_g = (ck - cirroi.y + cirroi.radius) + ci * nr_h;

                                double gval = 0.0;
                                if (interp_qbs_lut(cur_lut, cur_lut_h, cur_lut_w,
                                                   x_tilda_bcoef, y_tilda_bcoef, gval)) {
                                    g_buffer[lind_g] = gval;
                                } else {
                                    g_buffer[lind_g] = 0.0;
                                }
                                gm += g_buffer[lind_g];
                            }
                        }
                    }
                    gm /= (double)cirroi.region.totalpoints;

                    double deltag_inv = 0.0;
                    for (int ci = 0; ci < nr_h; ci++) {
                        for (int cj = 0; cj < cirroi.region.noderange[ci]; cj += 2) {
                            int top = cirroi.region.nodelist[ci + cj * nr_h];
                            int bot = cirroi.region.nodelist[ci + (cj + 1) * nr_h];
                            for (int ck = top; ck <= bot; ck++) {
                                int lind_g = (ck - cirroi.y + cirroi.radius) + ci * nr_h;
                                double diff = g_buffer[lind_g] - gm;
                                deltag_inv += diff * diff;
                            }
                        }
                    }
                    deltag_inv = std::sqrt(deltag_inv);
                    if (deltag_inv <= LAMBDA) {
                        corr_result = 0.0;
                        break;
                    }
                    deltag_inv = 1.0 / deltag_inv;

                    std::fill(gradient.begin(), gradient.end(), 0.0);
                    corr_result = 0.0;
                    for (int ci = 0; ci < nr_h; ci++) {
                        int px = ci + cirroi.x - cirroi.radius;
                        for (int cj = 0; cj < cirroi.region.noderange[ci]; cj += 2) {
                            int top = cirroi.region.nodelist[ci + cj * nr_h];
                            int bot = cirroi.region.nodelist[ci + (cj + 1) * nr_h];
                            for (int ck = top; ck <= bot; ck++) {
                                int lind_f = ck + px * ref_h;
                                int lind_g = (ck - cirroi.y + cirroi.radius) + ci * nr_h;
                                int lind_df = (ck - cirroi.y + cirroi.radius) * 6 + ci * (nr_h * 6);

                                double norm_diff = (ref_img[lind_f] - fm) * deltaf_inv
                                                   - (g_buffer[lind_g] - gm) * deltag_inv;

                                gradient[0] += norm_diff * df_dp_buffer[lind_df];
                                gradient[1] += norm_diff * df_dp_buffer[lind_df + 1];
                                gradient[2] += norm_diff * df_dp_buffer[lind_df + 2];
                                gradient[3] += norm_diff * df_dp_buffer[lind_df + 3];
                                gradient[4] += norm_diff * df_dp_buffer[lind_df + 4];
                                gradient[5] += norm_diff * df_dp_buffer[lind_df + 5];

                                corr_result += norm_diff * norm_diff;
                            }
                        }
                    }

                    for (int gi = 0; gi < 6; gi++) {
                        gradient[gi] *= 2.0 * deltaf_inv;
                    }

                    forwardsub(gradient.data(), hessian.data(), 6);
                    backwardsub(gradient.data(), hessian.data(), 6);
                    for (int gi = 0; gi < 6; gi++) {
                        gradient[gi] = -gradient[gi];
                    }

                    diffnorm = 0.0;
                    for (int gi = 0; gi < 6; gi++) {
                        diffnorm += gradient[gi] * gradient[gi];
                    }
                    diffnorm = std::sqrt(diffnorm);

                    double du = defvec[0], dv = defvec[1];
                    double dudx = defvec[2], dudy = defvec[3];
                    double dvdx = defvec[4], dvdy = defvec[5];
                    double dp0 = gradient[0], dp1 = gradient[1], dp2 = gradient[2];
                    double dp3 = gradient[3], dp4 = gradient[4], dp5 = gradient[5];

                    double denom = dp2 + dp5 + dp2 * dp5 - dp3 * dp4 + 1.0;

                    defvec[0] = du - ((dudx + 1.0) * (dp0 + dp0 * dp5 - dp1 * dp3)) / denom
                                  - (dudy * (dp1 - dp0 * dp4 + dp1 * dp2)) / denom;
                    defvec[1] = dv - ((dvdy + 1.0) * (dp1 - dp0 * dp4 + dp1 * dp2)) / denom
                                  - (dvdx * (dp0 + dp0 * dp5 - dp1 * dp3)) / denom;
                    defvec[2] = ((dp5 + 1.0) * (dudx + 1.0)) / denom
                                - (dp4 * dudy) / denom - 1.0;
                    defvec[3] = (dudy * (dp2 + 1.0)) / denom
                                - (dp3 * (dudx + 1.0)) / denom;
                    defvec[4] = (dvdx * (dp5 + 1.0)) / denom
                                - (dp4 * (dvdy + 1.0)) / denom;
                    defvec[5] = ((dp2 + 1.0) * (dvdy + 1.0)) / denom
                                - (dp3 * dvdx) / denom - 1.0;

                    if (diffnorm < cutoff_diffnorm) {
                        converged = true;
                        break;
                    }
                }

                calculated[nidx] = 1;

                bool accepted = converged &&
                    corr_result < cutoff_corrcoef &&
                    std::fabs(init_u - defvec[0]) < cutoff_disp &&
                    std::fabs(init_v - defvec[1]) < cutoff_disp &&
                    std::isfinite(defvec[0]) && std::isfinite(defvec[1]) &&
                    std::isfinite(defvec[2]) && std::isfinite(defvec[3]) &&
                    std::isfinite(defvec[4]) && std::isfinite(defvec[5]) &&
                    std::isfinite(corr_result) && diffnorm < 0.1;

                if (accepted) {
                    out_valid[nidx] = 1;

                    QueueItem next;
                    next.corrcoef = corr_result;
                    next.grid_x = gnx;
                    next.grid_y = gny;
                    next.u = defvec[0];
                    next.v = defvec[1];
                    next.du_dx = defvec[2];
                    next.du_dy = defvec[3];
                    next.dv_dx = defvec[4];
                    next.dv_dy = defvec[5];
                    queue.push(next);
                    num_computed++;
                } else {
                    double su = 0.0, sv = 0.0, sdudx = 0.0, sdudy = 0.0;
                    double sdvdx = 0.0, sdvdy = 0.0, scorr = 0.0, sdiff = 0.0;
                    int siter = 0;
                    int sret = calc_seed_point(
                        ref_img, ref_h, ref_w,
                        ref_grad_x, ref_grad_y, grad_h, grad_w,
                        border_bcoef,
                        cur_lut, cur_lut_h, cur_lut_w,
                        cur_img, cur_h, cur_w,
                        regions, num_regions,
                        nx, ny, radius, cutoff_diffnorm, cutoff_iteration, subsettrunc,
                        &su, &sv, &sdudx, &sdudy, &sdvdx, &sdvdy,
                        &scorr, &sdiff, &siter);
                    if (sret == 1 &&
                        scorr < 0.5 &&
                        sdiff <= 0.1 &&
                        std::fabs(su) <= radius + 1.0 &&
                        std::fabs(sv) <= radius + 1.0 &&
                        std::isfinite(su) && std::isfinite(sv) &&
                        std::isfinite(sdudx) && std::isfinite(sdudy) &&
                        std::isfinite(sdvdx) && std::isfinite(sdvdy) &&
                        std::isfinite(scorr)) {
                        out_valid[nidx] = 1;
                        out_u[nidx] = su;
                        out_v[nidx] = sv;
                        out_corrcoef[nidx] = scorr;
                        num_computed++;
                    }
                }
            }
        }
    }

    *out_points_computed = num_computed;
    return 1;
}

} // extern "C"
