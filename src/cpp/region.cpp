#include "types.hpp"
#include <algorithm>
#include <cmath>
#include <vector>
#include <cstdint>

namespace {

struct queue_roi_separate {
    int node_top;
    int node_bottom;
    int index;
};

struct roi_separate_buf {
    std::vector<std::vector<int>> nodelist;
    int leftbound;
    int rightbound;
    int totalpoints;
};

void form_regions_impl(std::vector<Region>& regions, bool& removed,
                       const uint8_t* mask, int mask_w, int mask_h,
                       int cutoff, bool preservelength);

} // namespace

// ============================================================
// form_boundary: 8-direction boundary trace
// ============================================================
extern "C"
void form_boundary(std::vector<std::vector<int>>& boundary,
                   int x_start, int y_start,
                   const uint8_t* mask, int mask_w, int mask_h,
                   int& direc) {
    boundary.clear();
    boundary.push_back({x_start, y_start});
    int newx = -1;
    int newy = -1;
    while (newx != boundary[0][0] || newy != boundary[0][1]) {
        direc = (direc + 6) % 8;
        int i;
        for (i = 0; i < 9; i++) {
            int x_inc = 0;
            int y_inc = 0;
            if (direc == 0) {
                x_inc = 1; y_inc = 0;
            } else if (direc == 1) {
                x_inc = 1; y_inc = -1;
            } else if (direc == 2) {
                x_inc = 0; y_inc = -1;
            } else if (direc == 3) {
                x_inc = -1; y_inc = -1;
            } else if (direc == 4) {
                x_inc = -1; y_inc = 0;
            } else if (direc == 5) {
                x_inc = -1; y_inc = 1;
            } else if (direc == 6) {
                x_inc = 0; y_inc = 1;
            } else {
                x_inc = 1; y_inc = 1;
            }
            newx = boundary.back()[0] + x_inc;
            newy = boundary.back()[1] + y_inc;
            if (newx >= 0 && newy >= 0 && newx < mask_w && newy < mask_h &&
                mask[newy + newx * mask_h]) {
                boundary.push_back({newx, newy});
                break;
            } else {
                direc = (direc + 1) % 8;
            }
        }
        if (i == 9) break;
    }
}

// ============================================================
// form_regions: 4-way connected component extraction
// ============================================================
extern "C"
void form_regions(std::vector<Region>& regions, bool& removed,
                  const uint8_t* mask, int mask_w, int mask_h,
                  int cutoff) {
    form_regions_impl(regions, removed, mask, mask_w, mask_h, cutoff, false);
}

// ============================================================
// within_region: check if point is in a region
// ============================================================
extern "C"
bool within_region(const Region& region, int x, int y) {
    int idx_x = x - region.leftbound;
    if (idx_x < 0 || idx_x >= region.height_nodelist) return false;
    for (int i = 0; i < region.noderange[idx_x]; i += 2) {
        if (y >= region.nodelist[idx_x + i * region.height_nodelist] &&
            y <= region.nodelist[idx_x + (i + 1) * region.height_nodelist]) {
            return true;
        }
    }
    return false;
}

// ============================================================
// get_cirroi: extract circular ROI subset from a region
// ============================================================
extern "C"
void get_cirroi(Cirroi& cirroi, const Region& region,
                int x, int y, int radius, bool subsettrunc) {
    int diam = 2 * radius + 1;
    int max_nodewidth = region.width_nodelist;

    // ---- allocate local state ----
    std::vector<int> circletemplate(diam * 2);           // (2R+1) x 2
    std::vector<uint8_t> cir_mask(diam * diam, 0);       // (2R+1) x (2R+1)
    std::vector<uint8_t> mask_buffer(diam * diam, 0);    // backup for trunc
    std::vector<int> queue_buffer(max_nodewidth);        // for insertion
    std::vector<int> queue_nodelist;
    std::vector<int> queue_nodeindex;
    queue_nodelist.reserve(max_nodewidth * diam);
    queue_nodeindex.reserve(max_nodewidth * diam / 2);

    std::vector<int> cir_nodelist(diam * max_nodewidth, 0);
    std::vector<int32_t> cir_noderange(diam, 0);
    int cir_totalpoints = 0;

    // activelines for source region: one flag per node pair
    // stored as activelines[col + pair_idx * region.height_nodelist]
    int num_pairs_per_col = max_nodewidth / 2;
    std::vector<uint8_t> activelines(region.height_nodelist * num_pairs_per_col, 1);

    // ---- build circle template ----
    // offset columns [0..2R], top = y + ceil(-sqrt(R^2-(col-R)^2))
    for (int col = 0; col < diam; col++) {
        int h = col - radius;
        double rad_sq = (double)radius * radius - (double)h * h;
        int top_off  = (int)ceil(-std::sqrt(rad_sq));
        int bot_off  = (int)floor(std::sqrt(rad_sq));
        circletemplate[col]         = y + top_off;
        circletemplate[col + diam]  = y + bot_off;
    }

    // ---- find center position in source region ----
    int idx_roi_x = x - region.leftbound;
    int idx_nodelist = -1;
    if (idx_roi_x >= 0 && idx_roi_x < region.height_nodelist) {
        for (int i = 0; i < region.noderange[idx_roi_x]; i += 2) {
            if (y >= region.nodelist[idx_roi_x + i * region.height_nodelist] &&
                y <= region.nodelist[idx_roi_x + (i + 1) * region.height_nodelist]) {
                idx_nodelist = i;
                break;
            }
        }
    }
    if (idx_nodelist == -1) return;

    // ---- initial node pair from center column ----
    bool circ_untrunc = true;
    int node_top, node_bottom;
    if (region.nodelist[idx_roi_x + idx_nodelist * region.height_nodelist] <
        circletemplate[radius]) {
        node_top = circletemplate[radius];
    } else {
        node_top = region.nodelist[idx_roi_x + idx_nodelist * region.height_nodelist];
        circ_untrunc = false;
    }
    if (region.nodelist[idx_roi_x + (idx_nodelist + 1) * region.height_nodelist] >
        circletemplate[radius + diam]) {
        node_bottom = circletemplate[radius + diam];
    } else {
        node_bottom = region.nodelist[idx_roi_x + (idx_nodelist + 1) * region.height_nodelist];
        circ_untrunc = false;
    }

    // update mask center column
    for (int row = node_top - (y - radius);
         row <= node_bottom - (y - radius); row++) {
        cir_mask[row + radius * diam] = 1;
    }

    // push center to queue
    queue_nodelist.push_back(node_top);
    queue_nodelist.push_back(node_bottom);
    queue_nodeindex.push_back(radius);

    // inactivate source node pair
    activelines[idx_roi_x + (idx_nodelist / 2) * region.height_nodelist] = 0;

    // ---- BFS queue expansion ----
    while (!queue_nodelist.empty()) {
        // pop bottom, top, index
        int q_bottom = queue_nodelist.back(); queue_nodelist.pop_back();
        int q_top    = queue_nodelist.back(); queue_nodelist.pop_back();
        int q_index  = queue_nodeindex.back(); queue_nodeindex.pop_back();

        // ---- insert into cirroi nodelist (sorted) ----
        if (cir_noderange[q_index] == 0) {
            cir_nodelist[q_index] = q_top;
            cir_nodelist[q_index + diam] = q_bottom;
        } else {
            bool inserted = false;
            int range = cir_noderange[q_index];
            for (int i = 0; i < range; i += 2) {
                if (q_bottom < cir_nodelist[q_index + i * diam]) {
                    // shift elements i..range-1 into queue_buffer
                    for (int j = i; j < range; j++) {
                        queue_buffer[j - i] = cir_nodelist[q_index + j * diam];
                    }
                    // insert new pair
                    cir_nodelist[q_index + i * diam] = q_top;
                    cir_nodelist[q_index + (i + 1) * diam] = q_bottom;
                    // shift back from queue_buffer
                    for (int j = i + 2; j < range + 2; j++) {
                        cir_nodelist[q_index + j * diam] = queue_buffer[j - (i + 2)];
                    }
                    inserted = true;
                    break;
                }
            }
            if (!inserted) {
                // append at end
                cir_nodelist[q_index + range * diam] = q_top;
                cir_nodelist[q_index + (range + 1) * diam] = q_bottom;
            }
        }
        cir_noderange[q_index] += 2;

        // source column index for this cirroi column
        int idx_froi = q_index + (x - radius) - region.leftbound;

        // ---- check LEFT neighbour ----
        if (q_index - 1 >= 0 && idx_froi - 1 >= 0) {
            for (int i = 0; i < region.noderange[idx_froi - 1]; i += 2) {
                if (activelines[idx_froi - 1 + (i / 2) * region.height_nodelist] == 0)
                    continue;
                if (region.nodelist[(idx_froi - 1) + (i + 1) * region.height_nodelist] < q_top)
                    continue;
                if (region.nodelist[(idx_froi - 1) + i * region.height_nodelist] <= q_bottom &&
                    region.nodelist[(idx_froi - 1) + (i + 1) * region.height_nodelist] >= q_top) {

                    int ntop, nbot;
                    if (region.nodelist[(idx_froi - 1) + i * region.height_nodelist] <
                        circletemplate[q_index - 1]) {
                        ntop = circletemplate[q_index - 1];
                    } else {
                        ntop = region.nodelist[(idx_froi - 1) + i * region.height_nodelist];
                        circ_untrunc = false;
                    }
                    if (region.nodelist[(idx_froi - 1) + (i + 1) * region.height_nodelist] >
                        circletemplate[(q_index - 1) + diam]) {
                        nbot = circletemplate[(q_index - 1) + diam];
                    } else {
                        nbot = region.nodelist[(idx_froi - 1) + (i + 1) * region.height_nodelist];
                        circ_untrunc = false;
                    }

                    if (ntop > nbot || ntop > q_bottom || nbot < q_top)
                        continue;

                    queue_nodelist.push_back(ntop);
                    queue_nodelist.push_back(nbot);
                    queue_nodeindex.push_back(q_index - 1);

                    activelines[idx_froi - 1 + (i / 2) * region.height_nodelist] = 0;

                    for (int row = ntop - (y - radius);
                         row <= nbot - (y - radius); row++) {
                        cir_mask[row + (q_index - 1) * diam] = 1;
                    }
                } else {
                    break;
                }
            }
        }

        // ---- check RIGHT neighbour ----
        if (q_index + 1 <= 2 * radius &&
            idx_froi + 1 <= region.rightbound - region.leftbound) {
            for (int i = 0; i < region.noderange[idx_froi + 1]; i += 2) {
                if (activelines[idx_froi + 1 + (i / 2) * region.height_nodelist] == 0)
                    continue;
                if (region.nodelist[(idx_froi + 1) + (i + 1) * region.height_nodelist] < q_top)
                    continue;
                if (region.nodelist[(idx_froi + 1) + i * region.height_nodelist] <= q_bottom &&
                    region.nodelist[(idx_froi + 1) + (i + 1) * region.height_nodelist] >= q_top) {

                    int ntop, nbot;
                    if (region.nodelist[(idx_froi + 1) + i * region.height_nodelist] <
                        circletemplate[q_index + 1]) {
                        ntop = circletemplate[q_index + 1];
                    } else {
                        ntop = region.nodelist[(idx_froi + 1) + i * region.height_nodelist];
                        circ_untrunc = false;
                    }
                    if (region.nodelist[(idx_froi + 1) + (i + 1) * region.height_nodelist] >
                        circletemplate[(q_index + 1) + diam]) {
                        nbot = circletemplate[(q_index + 1) + diam];
                    } else {
                        nbot = region.nodelist[(idx_froi + 1) + (i + 1) * region.height_nodelist];
                        circ_untrunc = false;
                    }

                    if (ntop > nbot || ntop > q_bottom || nbot < q_top)
                        continue;

                    queue_nodelist.push_back(ntop);
                    queue_nodelist.push_back(nbot);
                    queue_nodeindex.push_back(q_index + 1);

                    activelines[idx_froi + 1 + (i / 2) * region.height_nodelist] = 0;

                    for (int row = ntop - (y - radius);
                         row <= nbot - (y - radius); row++) {
                        cir_mask[row + (q_index + 1) * diam] = 1;
                    }
                } else {
                    break;
                }
            }
        }

        cir_totalpoints += q_bottom - q_top + 1;
    }

    // ---- set bounds ----
    int cir_upperbound = y + radius + 1;
    int cir_lowerbound = y - radius - 1;
    for (int i = 0; i < diam; i++) {
        if (cir_noderange[i] > 0) {
            if (cir_upperbound > cir_nodelist[i]) {
                cir_upperbound = cir_nodelist[i];
            }
            if (cir_lowerbound < cir_nodelist[i + (cir_noderange[i] - 1) * diam]) {
                cir_lowerbound = cir_nodelist[i + (cir_noderange[i] - 1) * diam];
            }
        }
    }

    // check for empty columns
    for (int i = 0; i < diam; i++) {
        if (cir_noderange[i] == 0) {
            circ_untrunc = false;
            break;
        }
    }

    // ---- subset truncation ----
    if (subsettrunc && !circ_untrunc) {
        // copy mask to buffer
        for (int i = 0; i < diam * diam; i++) {
            mask_buffer[i] = cir_mask[i];
        }

        // find top-left point in mask
        std::vector<std::vector<int>> vec_boundary;
        int tlx = -1, tly = -1;
        for (int col = 0; col < diam; col++) {
            for (int row = 0; row < diam; row++) {
                if (cir_mask[row + col * diam]) {
                    tlx = col; tly = row;
                    break;
                }
            }
            if (tlx != -1) break;
        }
        if (tlx == -1) goto skip_trunc;

        // trace boundary
        {
            int direc = 7;
            form_boundary(vec_boundary, tlx, tly,
                          cir_mask.data(), diam, diam, direc);
        }

        if (vec_boundary.empty()) goto skip_trunc;

        // find closest boundary point to center (radius, radius)
        {
            int idx_min = 0;
            double val_min =
                (double)(vec_boundary[0][0] - radius) * (vec_boundary[0][0] - radius) +
                (double)(vec_boundary[0][1] - radius) * (vec_boundary[0][1] - radius);
            for (int i = 1; i < (int)vec_boundary.size(); i++) {
                double d = (double)(vec_boundary[i][0] - radius) *
                               (vec_boundary[i][0] - radius) +
                           (double)(vec_boundary[i][1] - radius) *
                               (vec_boundary[i][1] - radius);
                if (d < val_min) {
                    val_min = d;
                    idx_min = i;
                }
            }

            int n_pts = (int)vec_boundary.size();

            // only proceed if closest point is not on boundary
            if ((int)ceil(std::sqrt(val_min)) < radius) {
                std::vector<double> p0(2, 0.0);
                std::vector<double> p1(2, 0.0);

                int idx_space = 3;
                int idx_plus  = mod_pos(idx_min + idx_space, n_pts);
                int idx_minus = mod_pos(idx_min - idx_space, n_pts);

                double x_plus_f = 0, x_min_f = 0, x_minus_f = 0;
                double y_plus_f = 0, y_min_f = 0, y_minus_f = 0;
                int length_filt = 2;
                for (int i = -length_filt; i <= length_filt; i++) {
                    x_plus_f  += (double)vec_boundary[mod_pos(idx_plus + i, n_pts)][0];
                    x_min_f   += (double)vec_boundary[mod_pos(idx_min + i, n_pts)][0];
                    x_minus_f += (double)vec_boundary[mod_pos(idx_minus + i, n_pts)][0];
                    y_plus_f  += (double)vec_boundary[mod_pos(idx_plus + i, n_pts)][1];
                    y_min_f   += (double)vec_boundary[mod_pos(idx_min + i, n_pts)][1];
                    y_minus_f += (double)vec_boundary[mod_pos(idx_minus + i, n_pts)][1];
                }
                int denom = 2 * length_filt + 1;
                x_plus_f  /= denom;
                x_min_f   /= denom;
                x_minus_f /= denom;
                y_plus_f  /= denom;
                y_min_f   /= denom;
                y_minus_f /= denom;

                double dx_di_f  = (x_plus_f - x_minus_f) / (2 * idx_space);
                double d2x_di2_f = (x_plus_f - 2 * x_min_f + x_minus_f) /
                                   ((double)idx_space * idx_space);
                double dy_di_f  = (y_plus_f - y_minus_f) / (2 * idx_space);
                double d2y_di2_f = (y_plus_f - 2 * y_min_f + y_minus_f) /
                                   ((double)idx_space * idx_space);

                double numer = (-x_min_f * dx_di_f + (double)radius * dx_di_f) +
                               (-y_min_f * dy_di_f + (double)radius * dy_di_f);
                double denom2 = (dx_di_f * dx_di_f + x_min_f * d2x_di2_f -
                                 (double)radius * d2x_di2_f) +
                                (dy_di_f * dy_di_f + y_min_f * d2y_di2_f -
                                 (double)radius * d2y_di2_f);
                double deltai = numer / denom2;

                if (std::fabs(deltai) < idx_space) {
                    double x_i = x_min_f + dx_di_f * deltai +
                                 0.5 * d2x_di2_f * deltai * deltai;
                    double y_i = y_min_f + dy_di_f * deltai +
                                 0.5 * d2y_di2_f * deltai * deltai;
                    double dx_di_i = dx_di_f + d2x_di2_f * deltai;
                    double dy_di_i = dy_di_f + d2y_di2_f * deltai;
                    double stepsize = 0.5;
                    p0[0] = x_i - dx_di_i * stepsize;
                    p0[1] = y_i - dy_di_i * stepsize;
                    p1[0] = x_i + dx_di_i * stepsize;
                    p1[1] = y_i + dy_di_i * stepsize;
                } else {
                    p0[0] = x_minus_f; p0[1] = y_minus_f;
                    p1[0] = x_plus_f;  p1[1] = y_plus_f;
                }

                // determine which side: p_subset is inside the mask
                std::vector<double> p_subset(2, 0.0);
                if (vec_boundary[idx_min][0] == radius &&
                    vec_boundary[idx_min][1] == radius) {
                    int width_win = 1;
                    int counter = 0;
                    for (int wi = -width_win; wi <= width_win; wi++) {
                        int xm = vec_boundary[idx_min][0] + wi;
                        for (int wj = -width_win; wj <= width_win; wj++) {
                            int ym = vec_boundary[idx_min][1] + wj;
                            if (xm >= 0 && xm <= 2 * radius &&
                                ym >= 0 && ym <= 2 * radius &&
                                cir_mask[ym + xm * diam]) {
                                p_subset[0] += xm;
                                p_subset[1] += ym;
                                counter++;
                            }
                        }
                    }
                    p_subset[0] /= counter;
                    p_subset[1] /= counter;
                } else {
                    p_subset[0] = (double)radius;
                    p_subset[1] = (double)radius;
                }

                // normalize against line shift
                p_subset[0] += p0[0] - (double)vec_boundary[idx_min][0];
                p_subset[1] += p0[1] - (double)vec_boundary[idx_min][1];

                int sign_clear = -sign(
                    (p1[0] - p0[0]) * (p_subset[1] - p0[1]) -
                    (p_subset[0] - p0[0]) * (p1[1] - p0[1]));

                // clear mask_buffer on the "wrong" side
                for (int col = 0; col < diam; col++) {
                    int xx = col;
                    for (int j = 0; j < cir_noderange[col]; j += 2) {
                        for (int k = cir_nodelist[col + j * diam];
                             k <= cir_nodelist[col + (j + 1) * diam]; k++) {
                            int yy = k - (y - radius);
                            if (sign((p1[0] - p0[0]) * ((double)yy - p0[1]) -
                                     ((double)xx - p0[0]) * (p1[1] - p0[1])) ==
                                sign_clear) {
                                mask_buffer[yy + xx * diam] = 0;
                            }
                        }
                    }
                }

                // re-extract regions from mask_buffer
                {
                    std::vector<Region> region_buf;
                    bool removed_flag = false;
                    form_regions_impl(region_buf, removed_flag,
                                      mask_buffer.data(), diam, diam,
                                      0, true);
                    if (!region_buf.empty()) {
                        // find largest region
                        int idx_max = 0;
                        int val_max = region_buf[0].totalpoints;
                        for (int i = 1; i < (int)region_buf.size(); i++) {
                            if (region_buf[i].totalpoints > val_max) {
                                idx_max = i;
                                val_max = region_buf[i].totalpoints;
                            }
                        }

                        // rebuild cir_mask from largest region
                        for (int i = 0; i < diam * diam; i++) cir_mask[i] = 0;
                        for (int col = 0; col < region_buf[idx_max].height_nodelist; col++) {
                            for (int j = 0; j < region_buf[idx_max].noderange[col];
                                 j += 2) {
                                for (int k = region_buf[idx_max].nodelist[col + j * region_buf[idx_max].height_nodelist];
                                     k <= region_buf[idx_max].nodelist[col + (j + 1) * region_buf[idx_max].height_nodelist];
                                     k++) {
                                    cir_mask[k + col * diam] = 1;
                                }
                            }
                        }

                        // convert region to global coords
                        Region& rb = region_buf[idx_max];
                        int offset_y = y - radius;
                        int offset_x = x - radius;
                        for (int col = 0; col < rb.height_nodelist; col++) {
                            for (int j = 0; j < rb.noderange[col]; j++) {
                                rb.nodelist[col + j * rb.height_nodelist] += offset_y;
                            }
                        }
                        rb.leftbound += offset_x;
                        rb.rightbound += offset_x;
                        rb.upperbound += offset_y;
                        rb.lowerbound += offset_y;

                        // store into cir nodelist
                        cir_nodelist.assign(diam * max_nodewidth, 0);
                        cir_noderange.assign(diam, 0);
                        cir_totalpoints = rb.totalpoints;
                        cir_upperbound = rb.upperbound;
                        cir_lowerbound = rb.lowerbound;
                        for (int col = 0; col < rb.height_nodelist; col++) {
                            cir_noderange[col] = rb.noderange[col];
                            for (int j = 0; j < rb.noderange[col]; j++) {
                                cir_nodelist[col + j * diam] =
                                    rb.nodelist[col + j * rb.height_nodelist];
                            }
                        }
                    }
                }
            }
        }
    }

skip_trunc:

    // ---- pack output ----
    cirroi.region.height_nodelist = diam;
    cirroi.region.width_nodelist  = max_nodewidth;
    cirroi.region.nodelist.resize(diam * max_nodewidth);
    cirroi.region.noderange.resize(diam);
    for (int col = 0; col < diam; col++) {
        cirroi.region.noderange[col] = cir_noderange[col];
        for (int j = 0; j < cir_noderange[col]; j++) {
            cirroi.region.nodelist[col + j * diam] = cir_nodelist[col + j * diam];
        }
    }
    cirroi.region.leftbound   = x - radius;
    cirroi.region.rightbound  = x + radius;
    cirroi.region.upperbound  = cir_upperbound;
    cirroi.region.lowerbound  = cir_lowerbound;
    cirroi.region.totalpoints = cir_totalpoints;

    cirroi.mask.assign(cir_mask.begin(), cir_mask.end());
    cirroi.radius = radius;
    cirroi.x = x;
    cirroi.y = y;
}

// ============================================================
// Internal: form_regions_impl (full version with preservelength)
// ============================================================
namespace {

void form_regions_impl(std::vector<Region>& regions, bool& removed,
                       const uint8_t* mask, int mask_w, int mask_h,
                       int cutoff, bool preservelength) {
    removed = false;

    // ---- Form overall ROI ----
    int leftbound, rightbound;
    bool firstpoint = true;

    if (preservelength) {
        leftbound = 0;
        rightbound = mask_w - 1;
        for (int i = 0; i < mask_w; i++) {
            for (int j = 0; j < mask_h; j++) {
                if (mask[j + i * mask_h]) {
                    firstpoint = false;
                    break;
                }
            }
            if (!firstpoint) break;
        }
    } else {
        leftbound = mask_w - 1;
        rightbound = 0;
        for (int i = 0; i < mask_w; i++) {
            for (int j = 0; j < mask_h; j++) {
                if (firstpoint && mask[j + i * mask_h]) {
                    leftbound = i;
                    firstpoint = false;
                }
                if (mask[j + i * mask_h]) {
                    rightbound = i;
                    break;
                }
            }
        }
    }

    if (firstpoint) return;

    int num_cols = rightbound - leftbound + 1;
    std::vector<std::vector<int>> nodelist_overall(num_cols);
    std::vector<std::vector<uint8_t>> activelines_overall(num_cols);

    // Build node pairs per column
    for (int i = leftbound; i <= rightbound; i++) {
        int col_idx = i - leftbound;
        bool start = false;
        int node_top = 0, node_bottom = 0;
        for (int j = 0; j < mask_h; j++) {
            if (!start && mask[j + i * mask_h]) {
                start = true;
                node_top = j;
            }
            if (start && (!mask[j + i * mask_h] || j == mask_h - 1)) {
                start = false;
                if (j == mask_h - 1 && mask[j + i * mask_h]) {
                    node_bottom = j;
                } else {
                    node_bottom = j - 1;
                }
                nodelist_overall[col_idx].push_back(node_top);
                nodelist_overall[col_idx].push_back(node_bottom);
                activelines_overall[col_idx].push_back(1);
            }
        }
    }

    // ---- Separate regions (4-way contiguous) ----
    std::vector<roi_separate_buf> roi_separate;
    std::vector<queue_roi_separate> queue_roi;

    int col = -1;
    while (col < (int)nodelist_overall.size() - 1) {
        ++col;

        if (nodelist_overall[col].size() == 0) continue;

        // find an active node pair in this column
        int idx_node = 0;
        bool activenodes = false;
        for (int j = 0; j < (int)activelines_overall[col].size(); j++) {
            if (activelines_overall[col][j]) {
                idx_node = j * 2;
                activelines_overall[col][j] = 0;
                activenodes = true;
                break;
            }
        }
        if (!activenodes) continue;

        roi_separate_buf buf;
        buf.nodelist.resize(num_cols);
        buf.totalpoints = 0;
        if (preservelength) {
            buf.leftbound = leftbound;
            buf.rightbound = rightbound;
        } else {
            buf.leftbound = col + leftbound;
            buf.rightbound = 0;
        }

        queue_roi_separate qt;
        qt.node_top    = nodelist_overall[col][idx_node];
        qt.node_bottom = nodelist_overall[col][idx_node + 1];
        qt.index       = col;
        queue_roi.push_back(qt);

        while (!queue_roi.empty()) {
            queue_roi_separate qb = queue_roi.back();
            queue_roi.pop_back();

            // insert + sort
            auto& cur_nl = buf.nodelist[qb.index];
            cur_nl.push_back(qb.node_top);
            cur_nl.push_back(qb.node_bottom);
            std::sort(cur_nl.begin(), cur_nl.end());

            buf.totalpoints += qb.node_bottom - qb.node_top + 1;

            // LEFT
            if (qb.index > 0) {
                for (int j = 0; j < (int)nodelist_overall[qb.index - 1].size(); j += 2) {
                    if (nodelist_overall[qb.index - 1][j] > qb.node_bottom)
                        break;
                    if (!activelines_overall[qb.index - 1][j / 2])
                        continue;
                    if (nodelist_overall[qb.index - 1][j + 1] < qb.node_top)
                        continue;
                    qt.node_top    = nodelist_overall[qb.index - 1][j];
                    qt.node_bottom = nodelist_overall[qb.index - 1][j + 1];
                    qt.index       = qb.index - 1;
                    queue_roi.push_back(qt);
                    activelines_overall[qb.index - 1][j / 2] = 0;
                }
            }

            // RIGHT
            if (qb.index < (int)nodelist_overall.size() - 1) {
                for (int j = 0; j < (int)nodelist_overall[qb.index + 1].size(); j += 2) {
                    if (nodelist_overall[qb.index + 1][j] > qb.node_bottom)
                        break;
                    if (!activelines_overall[qb.index + 1][j / 2])
                        continue;
                    if (nodelist_overall[qb.index + 1][j + 1] < qb.node_top)
                        continue;
                    qt.node_top    = nodelist_overall[qb.index + 1][j];
                    qt.node_bottom = nodelist_overall[qb.index + 1][j + 1];
                    qt.index       = qb.index + 1;
                    queue_roi.push_back(qt);
                    activelines_overall[qb.index + 1][j / 2] = 0;
                }
            }

            if (!preservelength && qb.index > buf.rightbound) {
                buf.rightbound = qb.index;
            }
        }

        if (!preservelength) {
            buf.rightbound += leftbound;
        }

        --col; // recheck current column

        if (buf.totalpoints > cutoff) {
            roi_separate.push_back(buf);
        } else {
            removed = true;
        }
    }

    // ---- Finalize regions ----
    regions.resize(roi_separate.size());
    for (int i = 0; i < (int)roi_separate.size(); i++) {
        int maxnodes = 0;
        for (int j = 0; j < (int)roi_separate[i].nodelist.size(); j++) {
            if ((int)roi_separate[i].nodelist[j].size() > maxnodes) {
                maxnodes = (int)roi_separate[i].nodelist[j].size();
            }
        }

        regions[i].height_nodelist =
            roi_separate[i].rightbound - roi_separate[i].leftbound + 1;
        regions[i].width_nodelist = maxnodes;
        regions[i].nodelist.resize(regions[i].height_nodelist *
                                   regions[i].width_nodelist);
        regions[i].noderange.resize(regions[i].height_nodelist);

        regions[i].leftbound   = roi_separate[i].leftbound;
        regions[i].rightbound  = roi_separate[i].rightbound;
        regions[i].totalpoints = roi_separate[i].totalpoints;

        regions[i].upperbound = mask_h;
        regions[i].lowerbound = 0;

        int idx_x = 0;
        for (int j = 0; j < (int)roi_separate[i].nodelist.size(); j++) {
            if (preservelength || roi_separate[i].nodelist[j].size() > 0) {
                regions[i].noderange[idx_x] =
                    (int)roi_separate[i].nodelist[j].size();
                for (int k = 0; k < (int)roi_separate[i].nodelist[j].size(); k += 2) {
                    regions[i].nodelist[idx_x +
                                        k * regions[i].height_nodelist] =
                        roi_separate[i].nodelist[j][k];
                    regions[i].nodelist[idx_x +
                                        (k + 1) * regions[i].height_nodelist] =
                        roi_separate[i].nodelist[j][k + 1];

                    if (roi_separate[i].nodelist[j][k] < regions[i].upperbound) {
                        regions[i].upperbound = roi_separate[i].nodelist[j][k];
                    }
                    if (roi_separate[i].nodelist[j][k + 1] >
                        regions[i].lowerbound) {
                        regions[i].lowerbound =
                            roi_separate[i].nodelist[j][k + 1];
                    }
                }
                ++idx_x;
            }
        }
    }
}

} // namespace
