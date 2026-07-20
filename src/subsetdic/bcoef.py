"""B-spline coefficient computation via FFT deconvolution.

Ported from Ncorr's ncorr_class_img.form_bcoef().
Key fix: kernel is CIRCULARLY centered so that β(0) is at index 0
of the FFT kernel, matching the QK matrix phase.
"""

import numpy as np


def beta5_nth(x, n=0):
    """Quintic B-spline beta5(x) and its n-th derivative (n=0..5)."""
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x)

    coeffs = np.array([1, -6, 15, -20, 15, -6, 1], dtype=np.float64)
    shifts = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.float64)

    for c, s in zip(coeffs, shifts):
        arg = x + s
        if n == 0:
            result += c * np.maximum(arg, 0.0) ** 5
        elif n == 1:
            result += c * 5.0 * np.maximum(arg, 0.0) ** 4
        elif n == 2:
            result += c * 20.0 * np.maximum(arg, 0.0) ** 3
        elif n == 3:
            result += c * 60.0 * np.maximum(arg, 0.0) ** 2
        elif n == 4:
            result += c * 120.0 * np.maximum(arg, 0.0) ** 1
        elif n == 5:
            result += c * 120.0 * (arg > 0).astype(np.float64)

    return result / 120.0


def compute_bspline_coefficients(img: np.ndarray, border: int = 20) -> np.ndarray:
    """Compute quintic B-spline coefficients via FFT deconvolution.

    Uses circularly-centered kernel [β(0), β(1), β(2), 0,...,0, β(-2), β(-1)]
    matching the QK matrix phase.
    """
    img = np.asarray(img, dtype=np.float64)
    h, w = img.shape
    bh, bw = h + 2 * border, w + 2 * border

    padded = np.pad(img, ((border, border), (border, border)), mode='symmetric')

    x_sample = np.array([-2, -1, 0, 1, 2], dtype=np.float64)
    kernel = beta5_nth(x_sample, n=0)

    # Row deconvolution: kernel circularly centered
    kernel_row = np.zeros(bw, dtype=np.float64)
    kernel_row[0] = kernel[2]
    kernel_row[1] = kernel[3]
    kernel_row[2] = kernel[4]
    kernel_row[bw - 2] = kernel[0]
    kernel_row[bw - 1] = kernel[1]
    kernel_fft_row = np.fft.fft(kernel_row)

    for i in range(bh):
        row_fft = np.fft.fft(padded[i, :])
        padded[i, :] = np.real(np.fft.ifft(row_fft / kernel_fft_row))

    # Column deconvolution
    kernel_col = np.zeros(bh, dtype=np.float64)
    kernel_col[0] = kernel[2]
    kernel_col[1] = kernel[3]
    kernel_col[2] = kernel[4]
    kernel_col[bh - 2] = kernel[0]
    kernel_col[bh - 1] = kernel[1]
    kernel_fft_col = np.fft.fft(kernel_col)

    for i in range(bw):
        col_fft = np.fft.fft(padded[:, i])
        padded[:, i] = np.real(np.fft.ifft(col_fft / kernel_fft_col))

    return padded
