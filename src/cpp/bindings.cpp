#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <vector>
#include <cstring>

#include "types.hpp"

namespace py = pybind11;

extern "C" {
    // solver.cpp
    void cholesky(double* mat, bool& positivedef, int size_mat);
    void forwardsub(double* vec, const double* mat, int size_mat);
    void backwardsub(double* vec, const double* mat, int size_mat);

    // interpolation.cpp
    bool interp_qbs_lut(const double* lut, int lut_h, int lut_w,
                        double x_tilda, double y_tilda, double& interp);

    // region.cpp
    void form_regions(std::vector<Region>& regions, bool& removed,
                      const uint8_t* mask, int mask_w, int mask_h, int cutoff);
    void form_boundary(std::vector<std::vector<int>>& boundary,
                       int x_start, int y_start, const uint8_t* mask,
                       int mask_w, int mask_h, int& direc);
    void get_cirroi(Cirroi& cirroi, const Region& region,
                    int x, int y, int radius, bool subsettrunc);
    bool within_region(const Region& region, int x, int y);

    // seeds.cpp (new signature with precomputed LUT)
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

    // rgdic.cpp (new signature with precomputed LUT)
    int rgdic_analysis(
        const double* ref_img, int ref_h, int ref_w,
        const double* ref_grad_x, const double* ref_grad_y,
        int grad_h, int grad_w,
        int border_bcoef,
        const double* cur_lut, int cur_lut_h, int cur_lut_w,
        const double* cur_img, int cur_h, int cur_w,
        const Region* regions, int num_regions,
        const double* seed_params, const int* seed_xs, const int* seed_ys, int num_seeds,
        int radius, int space, double cutoff_diffnorm, int cutoff_iteration, int subsettrunc,
        int direct_seed_grid,
        double* out_u, double* out_v, double* out_corrcoef, uint8_t* out_valid,
        int out_h, int out_w, int* out_points_computed);

    // strain.cpp
    int compute_strain(
        const double* plot_u, const double* plot_v,
        const Region* regions, int num_regions,
        const uint8_t* valid_mask, int grid_h, int grid_w,
        int radius_strain, int space, int subsettrunc,
        double* out_dudx, double* out_dudy, double* out_dvdx, double* out_dvdy,
        uint8_t* out_valid);

    // extrapolate.cpp
    int extrapolate_data(double* plot_data, int grid_h, int grid_w,
                         const int* region_mask, int border);
}

// ============================================================
// Helper: parse Python region dicts into C++ Region vector
// ============================================================
static std::vector<Region> parse_regions(py::list regions_list) {
    int num = (int)py::len(regions_list);
    std::vector<Region> regions(num);
    for (int i = 0; i < num; i++) {
        py::dict d = regions_list[i];
        regions[i].height_nodelist = d["height_nodelist"].cast<int>();
        regions[i].width_nodelist = d["width_nodelist"].cast<int>();
        regions[i].upperbound = d["upperbound"].cast<int>();
        regions[i].lowerbound = d["lowerbound"].cast<int>();
        regions[i].leftbound = d["leftbound"].cast<int>();
        regions[i].rightbound = d["rightbound"].cast<int>();
        regions[i].totalpoints = d["totalpoints"].cast<int>();
        auto nl = d["nodelist"].cast<py::array_t<int32_t>>();
        auto nr = d["noderange"].cast<py::array_t<int32_t>>();
        regions[i].nodelist.assign((int32_t*)nl.request().ptr,
                                   (int32_t*)nl.request().ptr + nl.request().size);
        regions[i].noderange.assign((int32_t*)nr.request().ptr,
                                    (int32_t*)nr.request().ptr + nr.request().size);
    }
    return regions;
}


PYBIND11_MODULE(_core, m) {
    m.doc() = "Subset-DIC core algorithms (C++ implementation)";

    // ------------------------------------------------------------
    // Region operations
    // ------------------------------------------------------------
    m.def("form_regions", [](py::array_t<uint8_t> mask, int cutoff) {
        auto buf = mask.request();
        int mask_h = (int)buf.shape[0];
        int mask_w = (int)buf.shape[1];
        std::vector<Region> regions;
        bool removed = false;
        form_regions(regions, removed, (const uint8_t*)buf.ptr, mask_w, mask_h, cutoff);
        py::list result;
        for (auto& r : regions) {
            py::dict d;
            d["nodelist"] = py::array_t<int32_t>({r.height_nodelist * r.width_nodelist}, r.nodelist.data());
            d["noderange"] = py::array_t<int32_t>({r.height_nodelist}, r.noderange.data());
            d["height_nodelist"] = r.height_nodelist;
            d["width_nodelist"] = r.width_nodelist;
            d["upperbound"] = r.upperbound;
            d["lowerbound"] = r.lowerbound;
            d["leftbound"] = r.leftbound;
            d["rightbound"] = r.rightbound;
            d["totalpoints"] = r.totalpoints;
            result.append(d);
        }
        return py::make_tuple(result, removed);
    }, py::arg("mask"), py::arg("cutoff") = 20);

    // ------------------------------------------------------------
    // Seed point calculation (with precomputed LUT)
    // ------------------------------------------------------------
    m.def("calc_seed", [](py::array_t<double> ref_img,
                           py::array_t<double> ref_grad_x,
                           py::array_t<double> ref_grad_y,
                           int border_bcoef,
                           py::array_t<double> cur_lut,
                           int cur_lut_h, int cur_lut_w,
                           py::array_t<double> cur_img,
                           py::list regions_list,
                           int seed_x, int seed_y,
                           int radius, double cutoff_diffnorm, int cutoff_iteration, bool subsettrunc) {
        auto r_buf = ref_img.request();
        auto gx_buf = ref_grad_x.request();
        auto gy_buf = ref_grad_y.request();
        auto cl_buf = cur_lut.request();
        auto ci_buf = cur_img.request();

        auto regions = parse_regions(regions_list);

        double u, v, dudx, dudy, dvdx, dvdy, corrcoef, diffnorm;
        int iterations;
        int ret = calc_seed_point(
            (const double*)r_buf.ptr, (int)r_buf.shape[0], (int)r_buf.shape[1],
            (const double*)gx_buf.ptr, (const double*)gy_buf.ptr,
            (int)gx_buf.shape[0], (int)gx_buf.shape[1],
            border_bcoef,
            (const double*)cl_buf.ptr, cur_lut_h, cur_lut_w,
            (const double*)ci_buf.ptr, (int)ci_buf.shape[0], (int)ci_buf.shape[1],
            regions.data(), (int)regions.size(),
            seed_x, seed_y, radius, cutoff_diffnorm, cutoff_iteration, (int)subsettrunc,
            &u, &v, &dudx, &dudy, &dvdx, &dvdy, &corrcoef, &diffnorm, &iterations);

        py::dict result;
        result["success"] = (ret == 1);
        result["u"] = u;
        result["v"] = v;
        result["dudx"] = dudx;
        result["dudy"] = dudy;
        result["dvdx"] = dvdx;
        result["dvdy"] = dvdy;
        result["corrcoef"] = corrcoef;
        result["diffnorm"] = diffnorm;
        result["iterations"] = iterations;
        return result;
    });

    // ------------------------------------------------------------
    // RG-DIC Analysis (with precomputed LUT)
    // ------------------------------------------------------------
    m.def("rgdic", [](py::array_t<double> ref_img,
                       py::array_t<double> ref_grad_x,
                       py::array_t<double> ref_grad_y,
                       int border_bcoef,
                       py::array_t<double> cur_lut,
                       int cur_lut_h, int cur_lut_w,
                       py::array_t<double> cur_img,
                       py::list regions_list,
                       py::array_t<double> seed_params,
                       py::array_t<int32_t> seed_xs, py::array_t<int32_t> seed_ys,
                       int radius, int spacing, double cutoff_diffnorm, int cutoff_iteration,
                       bool subsettrunc,
                       bool direct_seed_grid,
                       int out_h, int out_w) {
        auto r_buf = ref_img.request();
        auto gx_buf = ref_grad_x.request();
        auto gy_buf = ref_grad_y.request();
        auto cl_buf = cur_lut.request();
        auto ci_buf = cur_img.request();

        auto regions = parse_regions(regions_list);
        int num_seeds = (int)seed_xs.request().size;

        auto u_arr = py::array_t<double>({out_h, out_w});
        auto v_arr = py::array_t<double>({out_h, out_w});
        auto cc_arr = py::array_t<double>({out_h, out_w});
        auto valid_arr = py::array_t<uint8_t>({out_h, out_w});

        std::memset(u_arr.request().ptr, 0, (size_t)out_h * out_w * sizeof(double));
        std::memset(v_arr.request().ptr, 0, (size_t)out_h * out_w * sizeof(double));
        std::memset(cc_arr.request().ptr, 0, (size_t)out_h * out_w * sizeof(double));
        std::memset(valid_arr.request().ptr, 0, (size_t)out_h * out_w);

        int points_computed = 0;
        int ret = rgdic_analysis(
            (const double*)r_buf.ptr, (int)r_buf.shape[0], (int)r_buf.shape[1],
            (const double*)gx_buf.ptr, (const double*)gy_buf.ptr,
            (int)gx_buf.shape[0], (int)gx_buf.shape[1],
            border_bcoef,
            (const double*)cl_buf.ptr, cur_lut_h, cur_lut_w,
            (const double*)ci_buf.ptr, (int)ci_buf.shape[0], (int)ci_buf.shape[1],
            regions.data(), (int)regions.size(),
            (const double*)seed_params.request().ptr,
            (const int32_t*)seed_xs.request().ptr,
            (const int32_t*)seed_ys.request().ptr, num_seeds,
            radius, spacing, cutoff_diffnorm, cutoff_iteration, (int)subsettrunc,
            (int)direct_seed_grid,
            (double*)u_arr.request().ptr, (double*)v_arr.request().ptr,
            (double*)cc_arr.request().ptr, (uint8_t*)valid_arr.request().ptr,
            out_h, out_w, &points_computed);

        py::dict result;
        result["success"] = (ret == 1);
        result["u"] = u_arr;
        result["v"] = v_arr;
        result["corrcoef"] = cc_arr;
        result["valid"] = valid_arr;
        result["points_computed"] = points_computed;
        return result;
    });

    // ------------------------------------------------------------
    // Strain computation
    // ------------------------------------------------------------
    m.def("compute_strain", [](py::array_t<double> u, py::array_t<double> v,
                               py::list regions_list,
                               py::array_t<uint8_t> valid_mask,
                               int radius_strain, int space, bool subsettrunc) {
        auto u_buf = u.request();
        auto v_buf = v.request();
        auto vm_buf = valid_mask.request();
        int gh = (int)u_buf.shape[0];
        int gw = (int)u_buf.shape[1];
        auto regions = parse_regions(regions_list);

        auto dudx = py::array_t<double>({gh, gw});
        auto dudy = py::array_t<double>({gh, gw});
        auto dvdx = py::array_t<double>({gh, gw});
        auto dvdy = py::array_t<double>({gh, gw});
        auto out_valid = py::array_t<uint8_t>({gh, gw});

        compute_strain(
            (const double*)u_buf.ptr, (const double*)v_buf.ptr,
            regions.data(), (int)regions.size(),
            (const uint8_t*)vm_buf.ptr, gh, gw,
            radius_strain, space, (int)subsettrunc,
            (double*)dudx.request().ptr, (double*)dudy.request().ptr,
            (double*)dvdx.request().ptr, (double*)dvdy.request().ptr,
            (uint8_t*)out_valid.request().ptr);

        py::dict result;
        result["dudx"] = dudx;
        result["dudy"] = dudy;
        result["dvdx"] = dvdx;
        result["dvdy"] = dvdy;
        result["valid"] = out_valid;
        return result;
    });

    // ------------------------------------------------------------
    // Extrapolation
    // ------------------------------------------------------------
    m.def("extrapolate", [](py::array_t<double> data,
                            py::array_t<int32_t> region_mask, int border) {
        auto db = data.request();
        int gh = (int)db.shape[0];
        int gw = (int)db.shape[1];
        extrapolate_data((double*)db.ptr, gh, gw,
                         (const int*)region_mask.request().ptr, border);
    });
}
