"""QK matrix generation and QK*B*QK^T LUT precomputation.

Uses quintic B-spline basis functions to generate the QK matrix,
then precomputes the full QK*B*QK^T lookup table for fast interpolation.
"""

import numpy as np
from math import factorial


def generate_QK() -> np.ndarray:
    """Generate 6x6 QK matrix from quintic B-spline basis.

    QK[n, k] = (-1)^n * beta5^(n)(x[k]) / n!
    where x = [-2, -1, 0, 1, 2, 3]

    Returns:
        (6, 6) float64 QK matrix.
    """
    x_samples = np.array([-2, -1, 0, 1, 2, 3], dtype=np.float64)

    def plus_power(y, p):
        return np.maximum(y, 0.0) ** p

    coeffs = np.array([1, -6, 15, -20, 15, -6, 1], dtype=np.float64)
    shifts = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.float64)

    QK = np.zeros((6, 6), dtype=np.float64)
    for n in range(6):
        result = np.zeros(6, dtype=np.float64)
        for c, s in zip(coeffs, shifts):
            arg = x_samples + s
            if n == 0:
                result += c * plus_power(arg, 5)
            elif n == 1:
                result += c * 5.0 * plus_power(arg, 4)
            elif n == 2:
                result += c * 20.0 * plus_power(arg, 3)
            elif n == 3:
                result += c * 60.0 * plus_power(arg, 2)
            elif n == 4:
                result += c * 120.0 * plus_power(arg, 1)
            elif n == 5:
                result += c * 120.0 * (arg > 0).astype(np.float64)
        QK[n, :] = ((-1) ** n) * (result / 120.0) / factorial(n)

    return QK


def precompute_qk_lut(bcoef: np.ndarray, QK: np.ndarray) -> np.ndarray:
    """Precompute QK*B*QK^T lookup table for a B-spline coefficient image.

    For each pixel in the valid region, takes the 6x6 window of B-spline
    coefficients and computes QK @ window @ QK.T → 36 values.

    Args:
        bcoef: B-spline coefficient array, shape (H+2*border, W+2*border).
        QK: 6x6 QK matrix.

    Returns:
        LUT array of shape (lut_h, lut_w, 36) where lut_h = bcoef.shape[0] - 5.
        LUT is stored such that for pixel(i,j):
            the 6x6 window starts at bcoef[i:i+6, j:j+6]
            and LUT[i, j, :] = (QK @ window @ QK.T).ravel()
    """
    bh, bw = bcoef.shape
    lut_h = bh - 5
    lut_w = bw - 5

    if lut_h <= 0 or lut_w <= 0:
        raise ValueError(f"B-coef array too small: {bcoef.shape}, need at least 6x6")

    # Build 4D array of all 6x6 windows using stride tricks
    # blocks[i, j, :, :] = bcoef[i:i+6, j:j+6]
    from numpy.lib.stride_tricks import as_strided
    shape = (lut_h, lut_w, 6, 6)
    strides = (bcoef.strides[0], bcoef.strides[1], bcoef.strides[0], bcoef.strides[1])
    blocks = as_strided(bcoef, shape=shape, strides=strides)

    # Compute: QK @ blocks @ QK.T for all positions
    # blocks: (H, W, 6, 6), QK: (6, 6)
    # tmp = blocks @ QK.T: (H, W, 6, 6)
    # result = QK @ tmp: (H, W, 6, 6) -> need to contract left
    # Using np.einsum is simplest:
    # result[i, j, m, n] = sum_{k,l} QK[m, k] * blocks[i, j, k, l] * QK[n, l]
    result = np.einsum('mk,ijkl,nl->ijmn', QK, blocks, QK, optimize=True)

    # Flatten the last 2 dims: (H, W, 36)
    lut = result.reshape(lut_h, lut_w, 36)

    return lut


def extract_gradients_from_lut(lut: np.ndarray) -> tuple:
    """Extract gradient maps from QK_B_QKT LUT.

    Gradient at pixel center (dx=0.5, dy=0.5) is computed from the
    interpolation convention y_powers @ entry @ x_powers:
        df/dx = sum_{m,n} yv[m] * entry[m,n] * xv_dx[n]
        df/dy = sum_{m,n} yv_dx[m] * entry[m,n] * xv[n]
    where:
        xv = [1, 0.5, 0.25, 0.125, 0.0625, 0.03125]
        xv_dx = [0, 1, 1, 0.75, 0.5, 0.3125]

    Args:
        lut: QK_B_QKT LUT, shape (lut_h, lut_w, 36).

    Returns:
        (grad_x, grad_y): each shape (lut_h, lut_w), float64.
    """
    xv = np.array([1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125], dtype=np.float64)
    xv_dx = np.array([0.0, 1.0, 1.0, 0.75, 0.5, 0.3125], dtype=np.float64)

    lut_6x6 = lut.reshape(lut.shape[0], lut.shape[1], 6, 6)

    # grad_x[i,j] = xv @ lut_6x6[i,j] @ xv_dx
    grad_x = np.einsum('m,ijmn,n->ij', xv, lut_6x6, xv_dx, optimize=True)

    # grad_y[i,j] = xv_dx @ lut_6x6[i,j] @ xv
    grad_y = np.einsum('m,ijmn,n->ij', xv_dx, lut_6x6, xv, optimize=True)

    return grad_x, grad_y
