"""Run DIC analysis on case images."""
import numpy as np, os
from PIL import Image
from subsetdic import SubsetDIC

def run_case(case_name, dic_params, border=5):
    case_dir = os.path.join(r"C:\02Project\Research\SubsetDIC\case", case_name)
    result_dir = os.path.join(case_dir, "result")
    os.makedirs(result_dir, exist_ok=True)

    ref_img = np.array(Image.open(os.path.join(case_dir, "001.bmp")))
    def_img = np.array(Image.open(os.path.join(case_dir, "002.bmp")))
    roi_raw = np.array(Image.open(os.path.join(case_dir, "003.bmp")))

    ref = ref_img.astype(np.float64)
    cur = def_img.astype(np.float64)
    roi = (roi_raw > 128).astype(np.uint8)

    print(f"Case '{case_name}': ref={ref.shape}, roi_active={roi.sum()}")

    cfg = {
        "dic": {
            "radius": dic_params["radius"],
            "spacing": dic_params.get("spacing", 1),
            "cutoff_diffnorm": 1e-6,
            "cutoff_iteration": dic_params.get("cutoff_iteration", 30),
            "subsettrunc": False,
            "direct_seed_grid": dic_params.get("direct_seed_grid", False),
        },
        "seeds": {
            "method": dic_params.get("seed_method", "sift"),
            "n_seeds": dic_params.get("n_seeds", 3),
            "manual_positions": dic_params.get("manual_positions", []),
        },
        "strain": {
            "enabled": dic_params.get("compute_strain", True),
            "radius": dic_params.get("strain_radius", 15),
        },
        "postprocess": {
            "enabled": dic_params.get("postprocess", True),
            "max_iterations": dic_params.get("postprocess_max_iterations", 8),
            "min_neighbors": dic_params.get("postprocess_min_neighbors", 3),
            "corrcoef_threshold": dic_params.get("postprocess_corrcoef_threshold", 2.0),
        },
        "border": {"bcoef": border, "interp": border, "extrap": border},
    }

    dic = SubsetDIC(cfg)
    result = dic.run(ref, cur, roi)

    if not result.success:
        print(f"  FAILED"); return

    valid = result.valid.astype(bool); nv = valid.sum()
    print(f"  Valid pts: {nv}")

    if nv > 0:
        uv = result.u[valid]; vv = result.v[valid]
        um, us = np.median(uv), uv.std(); vm, vs = np.median(vv), vv.std()
        inl = (np.abs(uv - um) < 3*us) & (np.abs(vv - vm) < 3*vs)
        ug = uv[inl]; vg = vv[inl]
        print(f"  u: median={np.median(ug):.4f}, mean={ug.mean():.4f}, std={ug.std():.4f}, inliers={inl.sum()}/{nv}")
        print(f"  v: median={np.median(vg):.4f}, mean={vg.mean():.4f}, std={vg.std():.4f}")

    # Save
    np.save(os.path.join(result_dir, "u.npy"), result.u)
    np.save(os.path.join(result_dir, "v.npy"), result.v)
    np.save(os.path.join(result_dir, "corrcoef.npy"), result.corrcoef)
    np.save(os.path.join(result_dir, "valid.npy"), result.valid)
    if result.dudx is not None:
        for name, arr in [("dudx", result.dudx), ("dudy", result.dudy),
                           ("dvdx", result.dvdx), ("dvdy", result.dvdy)]:
            np.save(os.path.join(result_dir, f"{name}.npy"), arr)

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 3, figsize=(18, 10))
        ax[0,0].imshow(ref, cmap='gray'); ax[0,0].set_title('Reference')
        ax[0,1].imshow(cur, cmap='gray'); ax[0,1].set_title('Deformed')
        ax[0,2].imshow(roi, cmap='gray'); ax[0,2].set_title('ROI')
        ush = result.u.copy(); ush[~valid] = np.nan
        vsh = result.v.copy(); vsh[~valid] = np.nan
        im1 = ax[1,0].imshow(ush, cmap='jet'); ax[1,0].set_title('U displacement'); plt.colorbar(im1, ax=ax[1,0])
        im2 = ax[1,1].imshow(vsh, cmap='jet'); ax[1,1].set_title('V displacement'); plt.colorbar(im2, ax=ax[1,1])
        im3 = ax[1,2].imshow(result.corrcoef, cmap='hot'); ax[1,2].set_title('Correlation'); plt.colorbar(im3, ax=ax[1,2])
        plt.tight_layout(); plt.savefig(os.path.join(result_dir, "overview.png"), dpi=120); plt.close()
        print(f"  Overview saved")
    except Exception as e:
        print(f"  Plot error: {e}")

    print(f"  Results -> {result_dir}/\n")


if __name__ == "__main__":
    run_case("ring", {
        "radius": 15, "spacing": 3, "strain_radius": 20, "n_seeds": 3,
        "cutoff_iteration": 50, "seed_method": "sift",
    })
    run_case("star", {
        "radius": 5, "spacing": 0, "strain_radius": 15, "n_seeds": 10,
        "cutoff_iteration": 50, "seed_method": "manual",
        "direct_seed_grid": True,
        "manual_positions": [(512,128),(256,128),(768,128),(128,64),(384,64),(640,64),(896,64),(128,192),(384,192),(640,192),(896,192)],
    })
