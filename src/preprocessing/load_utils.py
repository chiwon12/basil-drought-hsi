"""
Hyperspectral I/O helpers.

`apply_reflectance_correction` is the manuscript's per-pixel reflectance formula:
    R = (raw - dark) / (white - dark)
clipped to [0, 1].
"""
from __future__ import annotations

import numpy as np


def load_cube(path: str) -> np.ndarray:
    """Load a hyperspectral cube (H, W, B) saved as .npy."""
    return np.load(path)


def load_wavelengths(path: str) -> np.ndarray:
    """Load a 1-D wavelength array (length B) saved as .npy."""
    return np.load(path)


def load_envi_cube(hdr_path: str) -> np.ndarray:
    """Load a hyperspectral cube from an ENVI .hdr/.raw pair.

    Requires the `spectral` package. Returns a (H, W, B) float32 array.
    """
    from spectral import open_image
    return open_image(hdr_path).load().astype(np.float32)


def apply_reflectance_correction(raw_cube: np.ndarray,
                                 white_ref: np.ndarray,
                                 dark_ref: np.ndarray,
                                 epsilon: float = 1e-6) -> np.ndarray:
    """Per-pixel reflectance: R = (raw - dark) / (white - dark), clipped to [0, 1].

    Shapes must be broadcastable. Typically raw is (H, W, B) and the references
    are either the same shape or (1, 1, B) after spatial averaging.
    """
    corrected = (raw_cube - dark_ref) / (white_ref - dark_ref + epsilon)
    return np.clip(corrected, 0.0, 1.0).astype(np.float32)
