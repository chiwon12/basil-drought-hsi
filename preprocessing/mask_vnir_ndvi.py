"""
VNIR leaf mask via NDVI + largest-connected-component selection.

Pipeline:
  1) Apply Savitzky–Golay smoothing along the spectral axis (window=11, poly=2)
     to stabilize NDVI on noisy bands.
  2) NDVI = (R_870 - R_650) / (R_870 + R_650).
  3) Median filter (size=5) → threshold (default 0.12) → morphological closing.
  4) Keep only the largest connected component (the plate of leaves).

Output: binary (uint8) leaf mask matching the spatial shape of the input cube.
        Pixel value 1 = leaf, 0 = background.

Default I/O (relative to git_open/):
    input   : zenodo_data/reflectance_cubes/vnir/*.npy   (reflectance, float32, [0,1])
    output  : zenodo_data/mask_outputs/vnir/{cube_stem}_mask.npy
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk, remove_small_objects

from load_utils import load_cube, load_wavelengths
from vegetation_index import compute_ndvi

REPO = Path(__file__).resolve().parents[2]


def vnir_leaf_mask(cube: np.ndarray, wavelengths: np.ndarray,
                   threshold: float = 0.12,
                   nir_nm: float = 870.0, red_nm: float = 650.0) -> np.ndarray:
    """Return binary leaf mask (uint8) for one VNIR reflectance cube."""
    smooth_cube = savgol_filter(cube, window_length=11, polyorder=2, axis=2)

    ndvi = compute_ndvi(smooth_cube, wavelengths, nir_nm=nir_nm, red_nm=red_nm)
    filt = median_filter(ndvi, size=5)

    binary = (filt > threshold).astype(np.uint8)
    binary = closing(binary, disk(5))
    binary = remove_small_objects(binary.astype(bool), min_size=200).astype(np.uint8)

    lbl = label(binary)
    regs = regionprops(lbl)
    if not regs:
        return np.zeros_like(binary)

    largest_id = max(regs, key=lambda r: r.area).label
    return (lbl == largest_id).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default=None,
                    help="reflectance-corrected VNIR cube directory "
                         "(default: zenodo_data/reflectance_cubes/vnir/)")
    ap.add_argument("--out_dir", default=None,
                    help="mask output directory "
                         "(default: zenodo_data/mask_outputs/vnir/)")
    ap.add_argument("--wavelengths", required=True,
                    help="path to a 1D numpy array of VNIR band centers in nm")
    ap.add_argument("--threshold", type=float, default=0.12)
    args = ap.parse_args()

    in_dir = Path(args.in_dir) if args.in_dir else REPO / "zenodo_data" / "reflectance_cubes" / "vnir"
    out_dir = Path(args.out_dir) if args.out_dir else REPO / "zenodo_data" / "mask_outputs" / "vnir"
    out_dir.mkdir(parents=True, exist_ok=True)

    wl = load_wavelengths(args.wavelengths)
    cubes = sorted(in_dir.glob("*.npy"))
    print(f"[mask-vnir] {len(cubes)} cube(s) in {in_dir}  threshold={args.threshold}")
    for p in cubes:
        cube = load_cube(str(p))
        mask = vnir_leaf_mask(cube, wl, threshold=args.threshold)
        out_p = out_dir / f"{p.stem}_mask.npy"
        np.save(out_p, mask)
        pct = 100.0 * mask.mean()
        print(f"  {p.name} -> {out_p.name}  leaf={pct:.1f}%")


if __name__ == "__main__":
    main()
