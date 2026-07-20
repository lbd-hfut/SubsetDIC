#include <vector>
#include "types.hpp"

extern "C" {

int extrapolate_data(
    double* plot_data,
    int grid_h, int grid_w,
    const int* region_mask,
    int border)
{
    const int total = grid_h * grid_w;

    // mask: true = needs filling (region_mask==1), false = has data (region_mask==0 or ==2)
    std::vector<bool> mask_new(total, false);
    std::vector<bool> mask_old(total, false);

    for (int i = 0; i < total; i++) {
        if (region_mask[i] == 1) {
            mask_new[i] = true;
            mask_old[i] = true;
        }
    }

    // ============================================================
    // Phase 1 - Data expansion: grow valid data outward by 1 pixel
    //            at a time until all holes are filled
    // ============================================================
    int totalcounter = 1;
    while (totalcounter > 0) {
        totalcounter = 0;

        for (int x = 0; x < grid_w; x++) {
            for (int y = 0; y < grid_h; y++) {
                const int idx = y + x * grid_h;

                if (mask_old[idx]) {
                    double sum = 0.0;
                    int counter = 0;

                    // Up
                    if (y > 0) {
                        int nidx = (y - 1) + x * grid_h;
                        if (!mask_old[nidx]) {
                            sum += plot_data[nidx];
                            counter++;
                        }
                    }
                    // Down
                    if (y < grid_h - 1) {
                        int nidx = (y + 1) + x * grid_h;
                        if (!mask_old[nidx]) {
                            sum += plot_data[nidx];
                            counter++;
                        }
                    }
                    // Left
                    if (x > 0) {
                        int nidx = y + (x - 1) * grid_h;
                        if (!mask_old[nidx]) {
                            sum += plot_data[nidx];
                            counter++;
                        }
                    }
                    // Right
                    if (x < grid_w - 1) {
                        int nidx = y + (x + 1) * grid_h;
                        if (!mask_old[nidx]) {
                            sum += plot_data[nidx];
                            counter++;
                        }
                    }

                    if (counter > 0) {
                        plot_data[idx] = sum / (double)counter;
                        mask_new[idx] = false;
                        totalcounter++;
                    }
                }
            }
        }

        // Update mask_old from mask_new
        for (int x = 0; x < grid_w; x++) {
            for (int y = 0; y < grid_h; y++) {
                const int idx = y + x * grid_h;
                mask_old[idx] = mask_new[idx];
            }
        }
    }

    // ============================================================
    // Phase 2 - Average filtering (FILTERITERATIONS passes)
    // ============================================================
    std::vector<double> paint_buffer(total);
    for (int i = 0; i < total; i++) {
        paint_buffer[i] = plot_data[i];
    }

    for (int iter = 0; iter < FILTERITERATIONS; iter++) {
        for (int x = 0; x < grid_w; x++) {
            for (int y = 0; y < grid_h; y++) {
                const int idx = y + x * grid_h;

                if (region_mask[idx] == 1) {
                    double sum = 0.0;
                    int count = 0;

                    for (int dy = -1; dy <= 1; dy++) {
                        int ny = y + dy;
                        if (ny < 0 || ny >= grid_h) continue;
                        for (int dx = -1; dx <= 1; dx++) {
                            int nx = x + dx;
                            if (nx < 0 || nx >= grid_w) continue;
                            sum += plot_data[ny + nx * grid_h];
                            count++;
                        }
                    }

                    paint_buffer[idx] = sum / (double)count;
                }
            }
        }

        // Copy paint_buffer back to plot_data for extrapolated points
        for (int x = 0; x < grid_w; x++) {
            for (int y = 0; y < grid_h; y++) {
                const int idx = y + x * grid_h;
                if (region_mask[idx] == 1) {
                    plot_data[idx] = paint_buffer[idx];
                }
            }
        }
    }

    return 0;
}

} // extern "C"
