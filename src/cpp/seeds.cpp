#include "types.hpp"
#include <cmath>
#include <algorithm>

extern "C" {

bool interp_qbs_lut(const double* lut, int lut_h, int lut_w,
                    double x_tilda, double y_tilda, double& interp);
void cholesky(double* mat, bool& positivedef, int size_mat);
void forwardsub(double* vec, const double* mat, int size_mat);
void backwardsub(double* vec, const double* mat, int size_mat);
void get_cirroi(Cirroi& cirroi, const Region& region, int x, int y,
                int radius, bool subsettrunc);
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
    double* out_u, double* out_v,
    double* out_dudx, double* out_dudy,
    double* out_dvdx, double* out_dvdy,
    double* out_corrcoef, double* out_diffnorm, int* out_iterations
) {
    // ---- 1. Find region containing (seed_x, seed_y) ----
    int region_idx = -1;
    for (int r = 0; r < num_regions; r++) {
        if (within_region(regions[r], seed_x, seed_y)) {
            region_idx = r;
            break;
        }
    }
    if (region_idx < 0) {
        *out_u = 0.0; *out_v = 0.0;
        *out_dudx = 0.0; *out_dudy = 0.0;
        *out_dvdx = 0.0; *out_dvdy = 0.0;
        *out_corrcoef = 0.0; *out_diffnorm = 0.0;
        *out_iterations = 0;
        return 0;
    }

    // ---- 2. Extract circular ROI subset ----
    Cirroi cirroi;
    get_cirroi(cirroi, regions[region_idx], seed_x, seed_y, radius, subsettrunc != 0);
    if (cirroi.region.totalpoints == 0) {
        *out_u = 0.0; *out_v = 0.0;
        *out_dudx = 0.0; *out_dudy = 0.0;
        *out_dvdx = 0.0; *out_dvdy = 0.0;
        *out_corrcoef = 0.0; *out_diffnorm = 0.0;
        *out_iterations = 0;
        return 0;
    }

    int diam = 2 * radius + 1;
    int nr_h = cirroi.region.height_nodelist;

    // ---- 3. NCC initial guess in spatial domain (multi-level) ----
    int reduction_multigrid = (int)std::max(1.0, std::floor(radius / 20.0));
    int stride = reduction_multigrid + 1;

    int red_cur_h = (cur_h + stride - 1) / stride;
    int red_cur_w = (cur_w + stride - 1) / stride;
    int red_ref_h = (diam + stride - 1) / stride;
    int red_ref_w = red_ref_h;

    std::vector<double> red_cur(red_cur_h * red_cur_w, 0.0);
    for (int x = 0; x < red_cur_w; x++) {
        int sx = x * stride;
        for (int y = 0; y < red_cur_h; y++) {
            int sy = y * stride;
            red_cur[y + x * red_cur_h] = cur_img[sy + sx * cur_h];
        }
    }

    std::vector<double> red_ref(red_ref_h * red_ref_w, 0.0);
    std::vector<uint8_t> red_mask(red_ref_h * red_ref_w, 0);
    int red_ref_n = 0;
    for (int i = 0; i < red_ref_w; i++) {
        int si = i * stride;
        if (si >= diam) break;
        for (int j = 0; j < red_ref_h; j++) {
            int sj = j * stride;
            if (sj >= diam) break;
            if (cirroi.mask[sj + si * diam]) {
                int abs_x = si + cirroi.x - radius;
                int abs_y = sj + cirroi.y - radius;
                red_ref[j + i * red_ref_h] = ref_img[abs_y + abs_x * ref_h];
                red_mask[j + i * red_ref_h] = 1;
                red_ref_n++;
            }
        }
    }

    if (red_ref_n < 9) {
        *out_u = 0.0; *out_v = 0.0;
        *out_dudx = 0.0; *out_dudy = 0.0;
        *out_dvdx = 0.0; *out_dvdy = 0.0;
        *out_corrcoef = 0.0; *out_diffnorm = 0.0;
        *out_iterations = 0;
        return 0;
    }

    double ref_mean = 0.0;
    for (int i = 0; i < red_ref_h; i++) {
        for (int j = 0; j < red_ref_w * red_ref_h; j += red_ref_h) {
            int idx = i + j;
            if (red_mask[idx]) ref_mean += red_ref[idx];
        }
    }
    ref_mean /= (double)red_ref_n;

    std::vector<double> ref_norm(red_ref_h * red_ref_w, 0.0);
    double ref_sqsum = 0.0;
    for (int idx = 0; idx < red_ref_h * red_ref_w; idx++) {
        if (red_mask[idx]) {
            ref_norm[idx] = red_ref[idx] - ref_mean;
            ref_sqsum += ref_norm[idx] * ref_norm[idx];
        }
    }

    if (ref_sqsum <= LAMBDA) {
        *out_u = 0.0; *out_v = 0.0;
        *out_dudx = 0.0; *out_dudy = 0.0;
        *out_dvdx = 0.0; *out_dvdy = 0.0;
        *out_corrcoef = 0.0; *out_diffnorm = 0.0;
        *out_iterations = 0;
        return 0;
    }

    double ref_ncc_denom = std::sqrt(ref_sqsum);

    int reduced_seed_x = seed_x / stride;
    int reduced_seed_y = seed_y / stride;

    int search_range_coarse = radius / stride + 1;
    int search_min_x = std::max(0, reduced_seed_x - search_range_coarse);
    int search_max_x = std::min(red_cur_w - red_ref_w, reduced_seed_x + search_range_coarse);
    int search_min_y = std::max(0, reduced_seed_y - search_range_coarse);
    int search_max_y = std::min(red_cur_h - red_ref_h, reduced_seed_y + search_range_coarse);

    if (search_min_x >= red_cur_w || search_max_x < 0 ||
        search_min_y >= red_cur_h || search_max_y < 0) {
        search_min_x = 0;
        search_max_x = red_cur_w - red_ref_w;
        search_min_y = 0;
        search_max_y = red_cur_h - red_ref_h;
    }

    double best_ncc = -2.0;
    int best_du = 0, best_dv = 0;

    std::vector<double> g_win(red_ref_n, 0.0);

    for (int ru = search_min_x; ru <= search_max_x; ru++) {
        for (int rv = search_min_y; rv <= search_max_y; rv++) {
            double gm_val = 0.0;
            int cnt = 0;
            for (int i = 0; i < red_ref_w; i++) {
                int cx = ru + i;
                if (cx < 0 || cx >= red_cur_w) continue;
                for (int j = 0; j < red_ref_h; j++) {
                    if (!red_mask[j + i * red_ref_h]) continue;
                    int cy = rv + j;
                    if (cy < 0 || cy >= red_cur_h) continue;
                    double val = red_cur[cy + cx * red_cur_h];
                    g_win[cnt] = val;
                    gm_val += val;
                    cnt++;
                }
            }
            if (cnt < red_ref_n) continue;

            gm_val /= (double)cnt;

            double cross = 0.0;
            double g_sq = 0.0;
            int cnt2 = 0;
            for (int i = 0; i < red_ref_w; i++) {
                for (int j = 0; j < red_ref_h; j++) {
                    if (!red_mask[j + i * red_ref_h]) continue;
                    double fn = ref_norm[j + i * red_ref_h];
                    double gn = g_win[cnt2] - gm_val;
                    cross += fn * gn;
                    g_sq += gn * gn;
                    cnt2++;
                }
            }

            if (g_sq <= LAMBDA) continue;

            double ncc = cross / (ref_ncc_denom * std::sqrt(g_sq));
            if (ncc > best_ncc) {
                best_ncc = ncc;
                best_du = ru * stride - seed_x + radius;
                best_dv = rv * stride - seed_y + radius;
            }
        }
    }

    double u_init = (double)best_du;
    double v_init = (double)best_dv;

    // ---- Optional second NCC level: full-resolution search around coarse guess ----
    if (stride > 1) {
        double fm_full = 0.0;
        for (int ci = 0; ci < nr_h; ci++) {
            for (int cj = 0; cj < cirroi.region.noderange[ci]; cj += 2) {
                int top = cirroi.region.nodelist[ci + cj * nr_h];
                int bot = cirroi.region.nodelist[ci + (cj + 1) * nr_h];
                for (int ck = top; ck <= bot; ck++) {
                    fm_full += ref_img[ck + (ci + cirroi.x - radius) * ref_h];
                }
            }
        }
        fm_full /= (double)cirroi.region.totalpoints;

        double ref_full_sqsum = 0.0;
        std::vector<double> ref_full_norm(cirroi.region.totalpoints, 0.0);
        int rfn_idx = 0;
        for (int ci = 0; ci < nr_h; ci++) {
            for (int cj = 0; cj < cirroi.region.noderange[ci]; cj += 2) {
                int top = cirroi.region.nodelist[ci + cj * nr_h];
                int bot = cirroi.region.nodelist[ci + (cj + 1) * nr_h];
                for (int ck = top; ck <= bot; ck++) {
                    double fn = ref_img[ck + (ci + cirroi.x - radius) * ref_h] - fm_full;
                    ref_full_norm[rfn_idx] = fn;
                    ref_full_sqsum += fn * fn;
                    rfn_idx++;
                }
            }
        }

        if (ref_full_sqsum > LAMBDA) {
            double ref_full_ncc_denom = std::sqrt(ref_full_sqsum);

            int fine_range = stride;
            int fine_min_x = std::max(-radius, best_du - fine_range);
            int fine_max_x = std::min(radius, best_du + fine_range);
            int fine_min_y = std::max(-radius, best_dv - fine_range);
            int fine_max_y = std::min(radius, best_dv + fine_range);

            double best_ncc_fine = -2.0;
            int best_du_fine = best_du;
            int best_dv_fine = best_dv;

            for (int du_val = fine_min_x; du_val <= fine_max_x; du_val++) {
                for (int dv_val = fine_min_y; dv_val <= fine_max_y; dv_val++) {
                    double gm_val = 0.0;
                    int cnt = 0;

                    for (int ci = 0; ci < nr_h; ci++) {
                        int cx = seed_x + du_val + (ci - radius);
                        if (cx < 0 || cx >= cur_w) {
                            for (int cj = 0; cj < cirroi.region.noderange[ci]; cj += 2) {
                                cnt += cirroi.region.nodelist[ci + (cj + 1) * nr_h]
                                     - cirroi.region.nodelist[ci + cj * nr_h] + 1;
                            }
                            continue;
                        }
                        for (int cj = 0; cj < cirroi.region.noderange[ci]; cj += 2) {
                            int top = cirroi.region.nodelist[ci + cj * nr_h];
                            int bot = cirroi.region.nodelist[ci + (cj + 1) * nr_h];
                            for (int ck = top; ck <= bot; ck++) {
                                int cy = ck + dv_val;
                                if (cy < 0 || cy >= cur_h) { cnt++; continue; }
                                double val = cur_img[cy + cx * cur_h];
                                gm_val += val;
                                cnt++;
                            }
                        }
                    }
                    if (cnt == 0) continue;
                    gm_val /= (double)cnt;

                    double cross = 0.0;
                    double g_sq = 0.0;
                    int cnt2 = 0;

                    for (int ci = 0; ci < nr_h; ci++) {
                        int cx = seed_x + du_val + (ci - radius);
                        for (int cj = 0; cj < cirroi.region.noderange[ci]; cj += 2) {
                            int top = cirroi.region.nodelist[ci + cj * nr_h];
                            int bot = cirroi.region.nodelist[ci + (cj + 1) * nr_h];
                            for (int ck = top; ck <= bot; ck++) {
                                double fn = ref_full_norm[cnt2];
                                double gn = 0.0;
                                int cy_img = ck + dv_val;
                                if (cx >= 0 && cx < cur_w && cy_img >= 0 && cy_img < cur_h) {
                                    gn = cur_img[cy_img + cx * cur_h] - gm_val;
                                }
                                cross += fn * gn;
                                g_sq += gn * gn;
                                cnt2++;
                            }
                        }
                    }

                    if (g_sq <= LAMBDA) continue;

                    double ncc = cross / (ref_full_ncc_denom * std::sqrt(g_sq));
                    if (ncc > best_ncc_fine) {
                        best_ncc_fine = ncc;
                        best_du_fine = du_val;
                        best_dv_fine = dv_val;
                    }
                }
            }

            u_init = (double)best_du_fine;
            v_init = (double)best_dv_fine;
        }
    }

    // ---- 4. Reference subset statistics (full resolution) ----
    double fm = 0.0;
    for (int ci = 0; ci < nr_h; ci++) {
        for (int cj = 0; cj < cirroi.region.noderange[ci]; cj += 2) {
            int top = cirroi.region.nodelist[ci + cj * nr_h];
            int bot = cirroi.region.nodelist[ci + (cj + 1) * nr_h];
            for (int ck = top; ck <= bot; ck++) {
                fm += ref_img[ck + (ci + cirroi.x - radius) * ref_h];
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
                double diff = ref_img[ck + (ci + cirroi.x - radius) * ref_h] - fm;
                deltaf_inv += diff * diff;
            }
        }
    }
    deltaf_inv = std::sqrt(deltaf_inv);
    if (deltaf_inv <= LAMBDA) {
        *out_u = u_init; *out_v = v_init;
        *out_dudx = 0.0; *out_dudy = 0.0;
        *out_dvdx = 0.0; *out_dvdy = 0.0;
        *out_corrcoef = 0.0; *out_diffnorm = 0.0;
        *out_iterations = 0;
        return 0;
    }
    deltaf_inv = 1.0 / deltaf_inv;

    // ---- 5. Precompute steepest descent images from gradient maps ----
    int buf_pixels = nr_h * diam;
    std::vector<double> df_dp(buf_pixels * 6, 0.0);
    std::vector<uint8_t> valid_pixel(buf_pixels, 0);

    for (int ci = 0; ci < nr_h; ci++) {
        for (int cj = 0; cj < cirroi.region.noderange[ci]; cj += 2) {
            int top = cirroi.region.nodelist[ci + cj * nr_h];
            int bot = cirroi.region.nodelist[ci + (cj + 1) * nr_h];
            for (int ck = top; ck <= bot; ck++) {
                double dx = (double)(ci - radius);
                double dy = (double)(ck - cirroi.y);
                int y_floor = ck;
                int x_floor = ci + (cirroi.x - radius);

                int gidy = y_floor + (border_bcoef > 0 ? border_bcoef - 2 : 0);
                int gidx = x_floor + (border_bcoef > 0 ? border_bcoef - 2 : 0);

                int lidx = ((ck - cirroi.y) + cirroi.radius) + ci * nr_h;

                if (gidy >= 0 && gidx >= 0 && gidy < grad_h && gidx < grad_w) {
                    int gi = gidy + gidx * grad_h;
                    double dfdx = ref_grad_x[gi];
                    double dfdy = ref_grad_y[gi];

                    int l6 = lidx * 6;
                    df_dp[l6    ] = dfdx;
                    df_dp[l6 + 1] = dfdy;
                    df_dp[l6 + 2] = dfdx * dx;
                    df_dp[l6 + 3] = dfdx * dy;
                    df_dp[l6 + 4] = dfdy * dx;
                    df_dp[l6 + 5] = dfdy * dy;
                    valid_pixel[lidx] = 1;
                }
            }
        }
    }

    // ---- 6. Build 6x6 GN Hessian (lower triangle) ----
    double hessian[36] = {0};
    for (int ci = 0; ci < nr_h; ci++) {
        for (int cj = 0; cj < cirroi.region.noderange[ci]; cj += 2) {
            int top = cirroi.region.nodelist[ci + cj * nr_h];
            int bot = cirroi.region.nodelist[ci + (cj + 1) * nr_h];
            for (int ck = top; ck <= bot; ck++) {
                int lidx = ((ck - cirroi.y) + cirroi.radius) + ci * nr_h;
                if (!valid_pixel[lidx]) continue;
                int l6 = lidx * 6;
                double* d = &df_dp[l6];
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

    // Symmetrize and multiply
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

    // ---- 7. Cholesky decompose Hessian ----
    bool positivedef = true;
    cholesky(hessian, positivedef, 6);
    if (!positivedef) {
        *out_u = u_init; *out_v = v_init;
        *out_dudx = 0.0; *out_dudy = 0.0;
        *out_dvdx = 0.0; *out_dvdy = 0.0;
        *out_corrcoef = 0.0; *out_diffnorm = 0.0;
        *out_iterations = 0;
        return 0;
    }

    // ---- 8. Iterative IC-GN Newton steps ----
    double defvec[6] = {u_init, v_init, 0.0, 0.0, 0.0, 0.0};
    double corrcoef_result = 0.0;
    double diffnorm_result = 0.0;
    int iter_used = 0;
    bool converged = false;

    std::vector<double> g_buffer(buf_pixels, 0.0);
    std::vector<double> gradient(6, 0.0);

    for (int iter = 0; iter <= cutoff_iteration; iter++) {
        iter_used = iter;

        // 8a. Interpolate current image at warped positions via LUT
        double gm = 0.0;
        for (int ci = 0; ci < nr_h; ci++) {
            for (int cj = 0; cj < cirroi.region.noderange[ci]; cj += 2) {
                int top = cirroi.region.nodelist[ci + cj * nr_h];
                int bot = cirroi.region.nodelist[ci + (cj + 1) * nr_h];
                for (int ck = top; ck <= bot; ck++) {
                    double ldx = (double)(ci - radius);
                    double ldy = (double)(ck - cirroi.y);

                    double x_tilda = (double)(ci + cirroi.x - radius) +
                                     defvec[0] + defvec[2] * ldx + defvec[3] * ldy;
                    double y_tilda = (double)ck +
                                     defvec[1] + defvec[4] * ldx + defvec[5] * ldy;

                    double bx_tilda = x_tilda + (double)border_bcoef;
                    double by_tilda = y_tilda + (double)border_bcoef;

                    int lidx = ((ck - cirroi.y) + cirroi.radius) + ci * nr_h;

                    double gval = 0.0;
                    if (interp_qbs_lut(cur_lut, cur_lut_h, cur_lut_w,
                                       bx_tilda, by_tilda, gval)) {
                        g_buffer[lidx] = gval;
                        gm += gval;
                    } else {
                        g_buffer[lidx] = 0.0;
                    }
                }
            }
        }
        gm /= (double)cirroi.region.totalpoints;

        // 8b. Compute deltag_inv
        double deltag_inv = 0.0;
        for (int ci = 0; ci < nr_h; ci++) {
            for (int cj = 0; cj < cirroi.region.noderange[ci]; cj += 2) {
                int top = cirroi.region.nodelist[ci + cj * nr_h];
                int bot = cirroi.region.nodelist[ci + (cj + 1) * nr_h];
                for (int ck = top; ck <= bot; ck++) {
                    int lidx = (ck - cirroi.y) + cirroi.radius + ci * nr_h;
                    double diff = g_buffer[lidx] - gm;
                    deltag_inv += diff * diff;
                }
            }
        }
        deltag_inv = std::sqrt(deltag_inv);
        if (deltag_inv <= LAMBDA) {
            corrcoef_result = 0.0;
            break;
        }
        deltag_inv = 1.0 / deltag_inv;

        // 8c-8d. Compute normalized differences and gradient
        std::fill(gradient.begin(), gradient.end(), 0.0);
        corrcoef_result = 0.0;

        for (int ci = 0; ci < nr_h; ci++) {
            for (int cj = 0; cj < cirroi.region.noderange[ci]; cj += 2) {
                int top = cirroi.region.nodelist[ci + cj * nr_h];
                int bot = cirroi.region.nodelist[ci + (cj + 1) * nr_h];
                for (int ck = top; ck <= bot; ck++) {
                    int lidx = (ck - cirroi.y) + cirroi.radius + ci * nr_h;
                    if (!valid_pixel[lidx]) continue;
                    int l6 = lidx * 6;

                    int abs_x = ci + cirroi.x - radius;
                    double norm_diff = (ref_img[ck + abs_x * ref_h] - fm) * deltaf_inv
                                       - (g_buffer[lidx] - gm) * deltag_inv;

                    gradient[0] += norm_diff * df_dp[l6];
                    gradient[1] += norm_diff * df_dp[l6 + 1];
                    gradient[2] += norm_diff * df_dp[l6 + 2];
                    gradient[3] += norm_diff * df_dp[l6 + 3];
                    gradient[4] += norm_diff * df_dp[l6 + 4];
                    gradient[5] += norm_diff * df_dp[l6 + 5];

                    corrcoef_result += norm_diff * norm_diff;
                }
            }
        }

        for (int gi = 0; gi < 6; gi++) {
            gradient[gi] *= 2.0 * deltaf_inv;
        }

        // 8e. Solve linear system
        forwardsub(gradient.data(), hessian, 6);
        backwardsub(gradient.data(), hessian, 6);

        // 8f. Negate
        for (int gi = 0; gi < 6; gi++) {
            gradient[gi] = -gradient[gi];
        }

        // 8g. Compute diffnorm
        diffnorm_result = 0.0;
        for (int gi = 0; gi < 6; gi++) {
            diffnorm_result += gradient[gi] * gradient[gi];
        }
        diffnorm_result = std::sqrt(diffnorm_result);

        // 8h. Inverse composition update
        {
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
        }

        // 8i. Check convergence
        if (diffnorm_result < cutoff_diffnorm) {
            converged = true;
            break;
        }
    }

    // ---- 9. Return results ----
    if (!converged && iter_used == cutoff_iteration) {
        converged = true;
    }

    *out_u = defvec[0];
    *out_v = defvec[1];
    *out_dudx = defvec[2];
    *out_dudy = defvec[3];
    *out_dvdx = defvec[4];
    *out_dvdy = defvec[5];
    *out_corrcoef = corrcoef_result;
    *out_diffnorm = diffnorm_result;
    *out_iterations = iter_used;

    return converged ? 1 : 0;
}

} // extern "C"
