"""
Vegetation indices computed from hyperspectral cubes.

Each function expects:
    cube       : (H, W, B) float32 reflectance
    wavelengths: (B,)     band centers in nm

Indices used in the manuscript:
    NDVI — leaf detection (VNIR)
    NDWI — leaf-water content (SWIR)
"""
from __future__ import annotations

import numpy as np


def find_band_index(wavelengths: np.ndarray, target_nm: float) -> int:
    return int(np.argmin(np.abs(np.asarray(wavelengths) - target_nm)))


def compute_ndvi(cube: np.ndarray, wavelengths: np.ndarray,
                 nir_nm: float = 870.0, red_nm: float = 650.0) -> np.ndarray:
    nir = cube[..., find_band_index(wavelengths, nir_nm)].astype(np.float32)
    red = cube[..., find_band_index(wavelengths, red_nm)].astype(np.float32)
    return (nir - red) / (nir + red + 1e-6)


def compute_ndwi(cube: np.ndarray, wavelengths: np.ndarray,
                 nir_nm: float = 950.0, swir_nm: float = 1500.0) -> np.ndarray:
    nir = cube[..., find_band_index(wavelengths, nir_nm)].astype(np.float32)
    swir = cube[..., find_band_index(wavelengths, swir_nm)].astype(np.float32)
    return (nir - swir) / (nir + swir + 1e-6)


def compute_gndvi(cube: np.ndarray, wavelengths: np.ndarray,
                  nir_nm: float = 870.0, green_nm: float = 530.0) -> np.ndarray:
    nir = cube[..., find_band_index(wavelengths, nir_nm)].astype(np.float32)
    grn = cube[..., find_band_index(wavelengths, green_nm)].astype(np.float32)
    return (nir - grn) / (nir + grn + 1e-6)


def compute_pri(cube: np.ndarray, wavelengths: np.ndarray,
                r531_nm: float = 531.0, r570_nm: float = 570.0) -> np.ndarray:
    a = cube[..., find_band_index(wavelengths, r531_nm)].astype(np.float32)
    b = cube[..., find_band_index(wavelengths, r570_nm)].astype(np.float32)
    return (a - b) / (a + b + 1e-6)
