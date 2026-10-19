"""DIODE evaluation-mask refinement from the released PPD implementation.

Reference: https://github.com/gangweix/pixel-perfect-depth/blob/main/ppd/data/diode.py
"""

import numpy as np
from scipy import ndimage


DIODE_SOBEL_THRESHOLD = 0.3


def refine_diode_valid_mask(depth, valid_mask):
    """Filter raw metric GT with PPD's unnormalized Sobel L1 response.

    Keep the provided mask, finite depths in [0.6, 350] metres and pixels
    whose |Sobel_y(depth)| + |Sobel_x(depth)| does not exceed 0.3.
    Neither the GT values nor the provided mask are modified in place.
    """
    depth = np.asarray(depth)
    valid_mask = np.asarray(valid_mask)
    if depth.ndim != 2 or depth.shape != valid_mask.shape:
        raise ValueError("DIODE depth and validity mask must be matching 2D arrays")
    refined = valid_mask.astype(bool, copy=True)
    refined &= np.isfinite(depth) & (depth >= 0.6) & (depth <= 350.0)
    # Match scipy.ndimage.sobel defaults (including reflect boundary handling),
    # the raw input depth and strict > threshold used in the reference code.
    dy = ndimage.sobel(depth, axis=0)
    dx = ndimage.sobel(depth, axis=1)
    refined[np.abs(dx) + np.abs(dy) > DIODE_SOBEL_THRESHOLD] = False
    return refined
