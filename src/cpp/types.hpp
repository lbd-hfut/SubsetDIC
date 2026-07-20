#pragma once

#include <vector>
#include <cstdint>
#include <cmath>

#define LAMBDA 1e-10
#define FILTERITERATIONS 50

inline double ncorr_round(double r) {
    return (r > 0.0) ? std::floor(r + 0.5) : std::ceil(r - 0.5);
}

inline int sign(double r) {
    return (r > 0) ? 1 : (r < 0) ? -1 : 0;
}

inline int mod_pos(int i, int n) {
    return (i % n + n) % n;
}

// ============================================================
// Region: column-major nodelist format (compatible with Ncorr)
// Each column stores (top,bottom) node pairs.
// nodelist shape: [height_nodelist x width_nodelist], col-major.
// noderange[i] = number of node entries in column i (always even).
// ============================================================
struct Region {
    std::vector<int32_t> nodelist;
    std::vector<int32_t> noderange;
    int height_nodelist = 0;
    int width_nodelist = 0;
    int upperbound = 0;
    int lowerbound = 0;
    int leftbound = 0;
    int rightbound = 0;
    int totalpoints = 0;

    bool within(int x, int y) const {
        int idx_x = x - leftbound;
        if (idx_x < 0 || idx_x >= height_nodelist) return false;
        for (int i = 0; i < noderange[idx_x]; i += 2) {
            int top = nodelist[idx_x + i * height_nodelist];
            int bot = nodelist[idx_x + (i + 1) * height_nodelist];
            if (y >= top && y <= bot) return true;
        }
        return false;
    }
};

// ============================================================
// Cirroi: Circular ROI subset within a region
// ============================================================
struct Cirroi {
    Region region;
    std::vector<uint8_t> mask;  // (2*radius+1) x (2*radius+1), col-major
    int radius = 0;
    int x = 0;
    int y = 0;
};

enum class Outcome { SUCCESS = 1, FAILED = 0, CANCELLED = -1 };

// ============================================================
// SeedInfo: seed point parameters
// ============================================================
struct SeedInfo {
    double x, y;
    double u, v;
    double du_dx, du_dy, dv_dx, dv_dy;
    double corrcoef;
    int num_region;
    int num_thread;
    int computepoints;
};

// ============================================================
// DicResult: final output
// ============================================================
struct DicResult {
    std::vector<double> u;           // displacement u field, HxW
    std::vector<double> v;           // displacement v field, HxW
    std::vector<double> corrcoef;    // correlation coefficient, HxW
    std::vector<uint8_t> valid;      // valid points mask, HxW
    std::vector<double> du_dx;       // strain du/dx (if computed)
    std::vector<double> du_dy;
    std::vector<double> dv_dx;
    std::vector<double> dv_dy;
    int height = 0;
    int width = 0;
    Outcome status = Outcome::SUCCESS;
};
