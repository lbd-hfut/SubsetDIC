"""Integration test for SubsetDIC (bilinear mode)."""
import numpy as np
from subsetdic import SubsetDIC


def test_translation():
    np.random.seed(42)
    h, w = 120, 120

    ref = np.random.rand(h, w).astype(np.float64)
    krn = np.ones((3, 3)) / 9.0
    ref_pad = np.pad(ref, 1, mode='edge')
    ref_smooth = np.zeros_like(ref)
    for i in range(h):
        for j in range(w):
            ref_smooth[i, j] = np.sum(ref_pad[i:i+3, j:j+3] * krn)

    ref_crop = ref_smooth[10:110, 10:110].astype(np.float64)
    cur = np.roll(ref_crop, 2, axis=1).copy()

    margin = 30
    roi = np.zeros((100, 100), dtype=np.uint8)
    roi[margin:100-margin, margin:100-margin] = 1

    dic = SubsetDIC({
        "dic": {"radius": 20, "spacing": 1, "cutoff_diffnorm": 1e-6, "cutoff_iteration": 30, "subsettrunc": False},
        "seeds": {"method": "manual", "n_seeds": 1,
                  "manual_positions": [(50, 50)]},
        "strain": {"enabled": False},
        "border": {"bcoef": 10, "interp": 10, "extrap": 10},
    })

    result = dic.run(ref_crop, cur, roi)
    assert result.success, "DIC failed"
    valid = result.valid.astype(bool)
    n_valid = valid.sum()
    assert n_valid > 100, f"Only {n_valid} valid points"

    u_vals = result.u[valid]
    v_vals = result.v[valid]
    med_u = np.median(u_vals)
    med_v = np.median(v_vals)

    good = (np.abs(u_vals - 2.0) < 1.0) & (np.abs(v_vals) < 1.0)
    good_pct = good.sum() / n_valid * 100
    print(f"Test: {n_valid} valid, {good_pct:.0f}% good, median u={med_u:.3f}, v={med_v:.3f}")

    assert good_pct > 80, f"Only {good_pct:.0f}% good"
    assert abs(med_u - 2.0) < 0.5
    print("PASSED")


if __name__ == "__main__":
    test_translation()
    print("All tests passed!")
