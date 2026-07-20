"""SIFT-based automatic seed point selection for SubsetDIC."""

import numpy as np
from typing import List, Tuple, Optional

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def sift_seed_selection(
    ref_img: np.ndarray,
    def_img: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
    n_seeds: int = 1,
    ratio_thresh: float = 0.7,
    min_displacement: float = 5.0,
) -> List[Tuple[int, int, float, float]]:
    if not HAS_CV2:
        return []

    ref = _to_uint8(ref_img)
    cur = _to_uint8(def_img)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(ref, None)
    kp2, des2 = sift.detectAndCompute(cur, None)

    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return []

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            pt1 = kp1[m.queryIdx].pt
            pt2 = kp2[m.trainIdx].pt
            dx = pt2[0] - pt1[0]
            dy = pt2[1] - pt1[1]
            disp = np.sqrt(dx ** 2 + dy ** 2)
            if disp >= min_displacement:
                if roi_mask is None or roi_mask[int(pt1[1]), int(pt1[0])]:
                    good.append((int(pt1[0]), int(pt1[1]), dx, dy, disp))

    good.sort(key=lambda x: x[4], reverse=True)
    return [(x, y, u, v) for x, y, u, v, _ in good[:n_seeds]]


def _to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    img = np.asarray(img, dtype=np.float64)
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max() * 255.0
    return img.astype(np.uint8)
