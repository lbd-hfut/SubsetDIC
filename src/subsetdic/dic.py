"""Top-level API for SubsetDIC - bilinear interpolation version."""

import numpy as np
from typing import Optional, Union
from .config import load_config
from .sift_seeds import sift_seed_selection
from . import _core

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class DicResult:
    def __init__(self):
        self.u = self.v = self.corrcoef = self.valid = None
        self.dudx = self.dudy = self.dvdx = self.dvdy = None
        self.success = False
        self.points_computed = 0
        self.regions = []


class SubsetDIC:
    def __init__(self, config: Optional[Union[str, dict]] = None):
        self._cfg = load_config(config)

    @property
    def config(self) -> dict:
        return dict(self._cfg)

    def run(self, ref_img, def_img, roi_mask=None, **kwargs):
        cfg = load_config(self._cfg)
        _apply_kwargs(cfg, kwargs)

        ref = np.asarray(ref_img, dtype=np.float64)
        cur = np.asarray(def_img, dtype=np.float64)
        if ref.shape != cur.shape:
            raise ValueError(f"Shape mismatch: {ref.shape} vs {cur.shape}")

        h, w = ref.shape
        roi = np.ones((h, w), dtype=np.uint8) if roi_mask is None else np.asarray(roi_mask, dtype=np.uint8)

        border = cfg["border"]["bcoef"]

        # Fortran conversion
        ref_f = np.asfortranarray(ref)
        cur_f = np.asfortranarray(cur)
        roi_f = np.asfortranarray(roi)

        # Gradient: central diff on ref (Fortran)
        gx = np.zeros_like(ref_f); gy = np.zeros_like(ref_f)
        gx[1:-1, 1:-1] = (ref_f[1:-1, 2:] - ref_f[1:-1, :-2]) * 0.5
        gy[1:-1, 1:-1] = (ref_f[2:, 1:-1] - ref_f[:-2, 1:-1]) * 0.5

        # LUT = current image (for bilinear interpolation)
        cur_lut = cur_f.ravel(order='F')

        # Regions
        regions_raw, _ = _core.form_regions(roi_f, cutoff=20)
        regions = [_region_dict(r) for r in regions_raw]
        if not regions:
            r = DicResult(); r.success = False; return r

        # Seeds
        seeds = _get_seeds(ref_f, cur_f, regions, cfg)
        if not seeds:
            r = DicResult(); r.success = False; return r

        # Refine
        refined = _refine_seeds(ref_f, gx, gy, cur_lut, cur_f, regions, seeds, cfg)
        if not refined:
            r = DicResult(); r.success = False; return r

        # RGDIC
        step = max(1, cfg["dic"].get("spacing", 1))
        oh = int(np.ceil(h / step)); ow = int(np.ceil(w / step))

        sp = np.array([[s["u"], s["v"], s["dudx"], s["dudy"], s["dvdx"], s["dvdy"]]
                        for s in refined], dtype=np.float64, order="C")
        sx = np.array([s["x"] // step for s in refined], dtype=np.int32)
        sy = np.array([s["y"] // step for s in refined], dtype=np.int32)

        rc = cfg["dic"]
        rr = _core.rgdic(ref_f, gx, gy, 0,
                          cur_lut, h, w,
                          regions, sp, sx, sy,
                          rc["radius"], step, rc["cutoff_diffnorm"],
                          rc["cutoff_iteration"], int(rc["subsettrunc"]),
                          bool(rc.get("direct_seed_grid", False)), oh, ow)

        result = DicResult()
        result.success = rr["success"]
        result.u = np.asarray(rr["u"])
        result.v = np.asarray(rr["v"])
        result.corrcoef = np.asarray(rr["corrcoef"])
        result.valid = np.asarray(rr["valid"])
        result.points_computed = rr["points_computed"]
        result.regions = regions

        if cfg.get("postprocess", {}).get("enabled", True) and result.success:
            roi_grid = _make_roi_grid(roi, result.u.shape, step)
            _postprocess_displacement(result, roi_grid, cfg["postprocess"])

        if cfg["strain"].get("enabled", True) and result.success:
            sr = _core.compute_strain(result.u, result.v, regions, result.valid,
                                       cfg["strain"]["radius"], step, rc["subsettrunc"])
            result.dudx = np.asarray(sr["dudx"]); result.dudy = np.asarray(sr["dudy"])
            result.dvdx = np.asarray(sr["dvdx"]); result.dvdy = np.asarray(sr["dvdy"])
            result.valid = np.asarray(sr["valid"])

        return result


def _region_dict(r):
    return {k: r[k] for k in ["nodelist", "noderange", "height_nodelist",
            "width_nodelist", "upperbound", "lowerbound", "leftbound",
            "rightbound", "totalpoints"]}


def _get_seeds(ref, cur, regions, cfg):
    sc = cfg["seeds"]
    method = sc.get("method", "sift")
    n = sc.get("n_seeds", 1)
    seeds = []
    if method == "sift":
        seeds = sift_seed_selection(ref, cur, n_seeds=n*3, min_displacement=1.0)
    else:
        for x, y in sc.get("manual_positions", []):
            seeds.append((int(x), int(y), 0.0, 0.0))

    r = cfg["dic"].get("radius", 20)
    h, w = ref.shape
    valid = []
    for x, y, u, v in seeds:
        if r <= x < w-r and r <= y < h-r:
            for rg in regions:
                if _within_region(rg, x, y):
                    valid.append({"x": int(x), "y": int(y), "u_init": u, "v_init": v})
                    break
    cy, cx = h//2, w//2
    valid.sort(key=lambda s: (s["x"]-cx)**2 + (s["y"]-cy)**2)
    if len(valid) < n and regions:
        for x, y in _region_candidate_points(regions[0], r, w, h):
            valid.append({"x": x, "y": y, "u_init": 0.0, "v_init": 0.0})
            if len(valid) >= n:
                break
    return valid[:n]


def _within_region(region, x, y):
    idx_x = int(x) - region["leftbound"]
    if idx_x < 0 or idx_x >= region["height_nodelist"]:
        return False
    nodelist = region["nodelist"]
    for i in range(0, int(region["noderange"][idx_x]), 2):
        top = nodelist[idx_x + i * region["height_nodelist"]]
        bot = nodelist[idx_x + (i + 1) * region["height_nodelist"]]
        if top <= y <= bot:
            return True
    return False


def _region_candidate_points(region, radius, width, height):
    cx = (region["leftbound"] + region["rightbound"]) // 2
    columns = list(range(region["leftbound"], region["rightbound"] + 1))
    columns.sort(key=lambda x: abs(x - cx))
    for x in columns:
        if not (radius <= x < width - radius):
            continue
        idx_x = x - region["leftbound"]
        for i in range(0, int(region["noderange"][idx_x]), 2):
            top = max(int(region["nodelist"][idx_x + i * region["height_nodelist"]]), radius)
            bot = min(int(region["nodelist"][idx_x + (i + 1) * region["height_nodelist"]]), height - radius - 1)
            if top <= bot:
                yield int(x), (top + bot) // 2


def _make_roi_grid(roi, out_shape, step):
    oh, ow = out_shape
    h, w = roi.shape
    y_idx = np.minimum(np.arange(oh) * step, h - 1)
    x_idx = np.minimum(np.arange(ow) * step, w - 1)
    return roi[np.ix_(y_idx, x_idx)].astype(bool)


def _postprocess_displacement(result, roi_grid, cfg):
    if not HAS_CV2:
        return

    u = np.asarray(result.u, dtype=np.float64).copy()
    v = np.asarray(result.v, dtype=np.float64).copy()
    corrcoef = np.asarray(result.corrcoef, dtype=np.float64).copy()
    valid = np.asarray(result.valid, dtype=np.uint8).astype(bool)

    max_iter = int(cfg.get("max_iterations", 8))
    min_neighbors = int(cfg.get("min_neighbors", 3))
    corr_thr = float(cfg.get("corrcoef_threshold", 2.0))

    finite = np.isfinite(u) & np.isfinite(v) & np.isfinite(corrcoef)
    good = roi_grid & valid & finite & (corrcoef <= corr_thr)
    fill_target = roi_grid & ~good
    if not fill_target.any() or not good.any():
        result.valid = (good & roi_grid).astype(np.uint8)
        return

    kernel = np.ones((3, 3), dtype=np.float64)
    kernel[1, 1] = 0.0

    for _ in range(max_iter):
        good_f = good.astype(np.float64)
        count = cv2.filter2D(good_f, -1, kernel, borderType=cv2.BORDER_CONSTANT)

        sum_u = cv2.filter2D(np.where(good, u, 0.0), -1, kernel, borderType=cv2.BORDER_CONSTANT)
        sum_v = cv2.filter2D(np.where(good, v, 0.0), -1, kernel, borderType=cv2.BORDER_CONSTANT)
        sum_c = cv2.filter2D(np.where(good, corrcoef, 0.0), -1, kernel, borderType=cv2.BORDER_CONSTANT)

        fill = fill_target & (count >= min_neighbors)
        if not fill.any():
            break

        u[fill] = sum_u[fill] / count[fill]
        v[fill] = sum_v[fill] / count[fill]
        corrcoef[fill] = sum_c[fill] / count[fill]
        valid[fill] = True
        good[fill] = True
        fill_target[fill] = False

    result.u = u
    result.v = v
    result.corrcoef = corrcoef
    result.valid = (good & roi_grid).astype(np.uint8)
    result.points_computed = int(result.valid.sum())


def _refine_seeds(ref_f, gx, gy, cur_lut, cur_f, regions, seeds, cfg):
    rc = cfg["dic"]
    radius = rc["radius"]
    h, w = ref_f.shape
    refined = []
    for s in seeds:
        res = _core.calc_seed(ref_f, gx, gy, 0,
                              cur_lut, h, w, cur_f,
                              regions, s["x"], s["y"],
                              radius, rc["cutoff_diffnorm"],
                              rc["cutoff_iteration"], int(rc["subsettrunc"]))
        if (res["success"]
                and np.isfinite([res["u"], res["v"], res["dudx"], res["dudy"], res["dvdx"], res["dvdy"], res["corrcoef"]]).all()
                and res["diffnorm"] <= max(rc["cutoff_diffnorm"] * 10.0, 0.1)
                and res["corrcoef"] < 0.5
                and abs(res["u"]) <= radius + 1
                and abs(res["v"]) <= radius + 1):
            refined.append({"x": s["x"], "y": s["y"],
                            "u": res["u"], "v": res["v"],
                            "dudx": res["dudx"], "dudy": res["dudy"],
                            "dvdx": res["dvdx"], "dvdy": res["dvdy"],
                            "corrcoef": res["corrcoef"]})
    return refined


def _apply_kwargs(cfg, kwargs):
    m = {"radius": ("dic", "radius"), "spacing": ("dic", "spacing"),
         "cutoff_diffnorm": ("dic", "cutoff_diffnorm"), "cutoff_iteration": ("dic", "cutoff_iteration"),
         "subsettrunc": ("dic", "subsettrunc"), "direct_seed_grid": ("dic", "direct_seed_grid"),
         "n_seeds": ("seeds", "n_seeds"),
         "strain_radius": ("strain", "radius"), "compute_strain": ("strain", "enabled")}
    for kw, (sec, key) in m.items():
        if kw in kwargs: cfg[sec][key] = kwargs[kw]
