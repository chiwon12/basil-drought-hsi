"""
Per-class NDVI x REP feature distribution → CSV (KDE grid + raw pixels).

For every VNIR reflectance cube under the input directory, this script:
  1) computes NDVI = (R_800 - R_667) / (R_800 + R_667),
  2) computes REP (red-edge position) = the wavelength of the maximum spectral
     gradient inside [low, high] (default 680–750 nm),
  3) keeps only leaf pixels with NDVI > 0.3,
  4) groups the resulting (NDVI, REP) pairs by treatment class extracted from
     the filename, and
  5) saves the result as CSVs (no plots).

Outputs (under <out_dir>/):
    ndvi_rep_pixels.csv    long table of every kept pixel:
                             columns = [class, ndvi, rep]
    ndvi_rep_kde_grid.csv  Gaussian-KDE density evaluated on a regular
                             NDVI x REP grid for each class:
                             columns = [class, ndvi, rep, density]
    ndvi_rep_summary.csv   per-class summary (n_pixels, mean/std/p25/p50/p75
                             for both NDVI and REP)

Class is extracted by matching `_(WW|MD|DR|SD)_` in the filename, so this
works with both the renamed bundle (`vnir_WW_plant1-1_20250421.npy`) and
the original naming (`VN_WW_BS_1-1_20250421_*_cube.npy`).

Default I/O (relative to git_open/):
    input  : zenodo_data/cubes/  (VNIR cubes — files containing 'vnir' in name)
    output : results/ndvi_rep_kde/
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Helper modules live in preprocessing/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from load_utils import load_cube, load_wavelengths
from vegetation_index import find_band_index

REPO = Path(__file__).resolve().parents[2]
SENSOR = "vnir"  # NDVI + REP are VNIR features (400–1000 nm)

CLASS_PATTERN = re.compile(r"_(WW|MD|DR|SD)_")
CLASS_ORDER = ["WW", "MD", "DR", "SD"]


# --------------------------------------------------------------------------- #
# Feature extractors                                                          #
# --------------------------------------------------------------------------- #
def class_from_filename(name: str) -> str | None:
    m = CLASS_PATTERN.search(name)
    return m.group(1) if m else None


def compute_ndvi_map(cube: np.ndarray, wavelengths: np.ndarray,
                     red_nm: float = 667.0, nir_nm: float = 800.0) -> np.ndarray:
    red = cube[..., find_band_index(wavelengths, red_nm)].astype(np.float64)
    nir = cube[..., find_band_index(wavelengths, nir_nm)].astype(np.float64)
    return (nir - red) / (nir + red + 1e-8)


def compute_rep_map(cube: np.ndarray, wavelengths: np.ndarray,
                    low_nm: float = 680.0, high_nm: float = 750.0) -> np.ndarray:
    """Red-edge position: per-pixel argmax of the spectral gradient in [low, high]."""
    band_idx = np.where((wavelengths >= low_nm) & (wavelengths <= high_nm))[0]
    if band_idx.size < 3:
        raise ValueError(f"too few bands in [{low_nm}, {high_nm}] nm: {band_idx.size}")
    wl_sel = wavelengths[band_idx]

    sub = cube[..., band_idx].astype(np.float64)        # (H, W, k)
    grad = np.gradient(sub, wl_sel, axis=2)             # (H, W, k)
    argmax = np.argmax(grad, axis=2)                    # (H, W)
    return wl_sel[argmax]


# --------------------------------------------------------------------------- #
# Aggregation + saving                                                        #
# --------------------------------------------------------------------------- #
def kde_grid_csv(class_data: dict[str, tuple[np.ndarray, np.ndarray]],
                 ndvi_grid: np.ndarray, rep_grid: np.ndarray) -> pd.DataFrame:
    """Evaluate a 2D Gaussian KDE per class on the same NDVI x REP grid."""
    rows = []
    X, Y = np.meshgrid(ndvi_grid, rep_grid, indexing="ij")
    positions = np.vstack([X.ravel(), Y.ravel()])
    for cls in CLASS_ORDER:
        ndvi_vals, rep_vals = class_data.get(cls, (np.empty(0), np.empty(0)))
        if ndvi_vals.size < 2:
            continue
        kernel = gaussian_kde(np.vstack([ndvi_vals, rep_vals]))
        density = kernel(positions).reshape(X.shape)
        for i, n in enumerate(ndvi_grid):
            for j, r in enumerate(rep_grid):
                rows.append({"class": cls, "ndvi": float(n),
                             "rep": float(r), "density": float(density[i, j])})
    return pd.DataFrame(rows)


def summary_csv(class_data: dict[str, tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    rows = []
    for cls in CLASS_ORDER:
        ndvi_vals, rep_vals = class_data.get(cls, (np.empty(0), np.empty(0)))
        if ndvi_vals.size == 0:
            continue
        rows.append({
            "class":         cls,
            "n_pixels":      int(ndvi_vals.size),
            "ndvi_mean":     float(ndvi_vals.mean()),
            "ndvi_std":      float(ndvi_vals.std()),
            "ndvi_p25":      float(np.percentile(ndvi_vals, 25)),
            "ndvi_p50":      float(np.percentile(ndvi_vals, 50)),
            "ndvi_p75":      float(np.percentile(ndvi_vals, 75)),
            "rep_mean":      float(rep_vals.mean()),
            "rep_std":       float(rep_vals.std()),
            "rep_p25":       float(np.percentile(rep_vals, 25)),
            "rep_p50":       float(np.percentile(rep_vals, 50)),
            "rep_p75":       float(np.percentile(rep_vals, 75)),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default=None,
                    help="directory of VNIR reflectance cubes "
                         "(default: zenodo_data/cubes/, filtered to *vnir*.npy)")
    ap.add_argument("--wavelengths", default=None,
                    help="VNIR wavelengths .npy "
                         "(default: data/wavelengths/wavelengths_vnir.npy)")
    ap.add_argument("--out_dir", default=None,
                    help="output directory (default: results/ndvi_rep_kde/)")
    ap.add_argument("--ndvi_min", type=float, default=0.3,
                    help="keep only pixels with NDVI greater than this (default 0.3)")
    ap.add_argument("--max_pixels_per_class", type=int, default=200_000,
                    help="random-sample to at most this many pixels per class for "
                         "the KDE fit (default 200,000)")
    ap.add_argument("--grid_n_ndvi", type=int, default=100)
    ap.add_argument("--grid_n_rep", type=int, default=100)
    ap.add_argument("--save_pixels", action="store_true",
                    help="also save the full per-pixel CSV (can be large)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    in_dir = Path(args.in_dir) if args.in_dir else REPO / "zenodo_data" / "cubes"
    wl_path = (Path(args.wavelengths) if args.wavelengths
               else REPO / "data" / "wavelengths" / "wavelengths_vnir.npy")
    out_dir = Path(args.out_dir) if args.out_dir else REPO / "results" / "ndvi_rep_kde"
    out_dir.mkdir(parents=True, exist_ok=True)

    wavelengths = load_wavelengths(str(wl_path))

    cubes = [p for p in sorted(in_dir.glob("*.npy")) if SENSOR in p.name.lower()]
    print(f"[ndvi_rep_kde] {len(cubes)} VNIR cube(s) in {in_dir}")

    pixel_rows = []          # for the optional long-table CSV
    by_class: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {c: [] for c in CLASS_ORDER}

    for p in cubes:
        cls = class_from_filename(p.name)
        if cls is None:
            print(f"  skip {p.name} (no WW/MD/DR/SD match)")
            continue

        cube = load_cube(str(p))
        ndvi_map = compute_ndvi_map(cube, wavelengths)
        rep_map = compute_rep_map(cube, wavelengths)

        leaf = ndvi_map > args.ndvi_min
        ndvi_vals = ndvi_map[leaf].astype(np.float32)
        rep_vals = rep_map[leaf].astype(np.float32)
        if ndvi_vals.size == 0:
            print(f"  {p.name}: no leaf pixels above NDVI={args.ndvi_min}")
            continue

        by_class[cls].append((ndvi_vals, rep_vals))
        print(f"  {p.name}  class={cls}  leaf_pixels={ndvi_vals.size:,}")

        if args.save_pixels:
            pixel_rows.append(pd.DataFrame({
                "class": cls, "source": p.name, "ndvi": ndvi_vals, "rep": rep_vals,
            }))

    # concatenate + sub-sample per class
    rng = np.random.default_rng(args.seed)
    class_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cls, chunks in by_class.items():
        if not chunks:
            continue
        ndvi_all = np.concatenate([c[0] for c in chunks])
        rep_all = np.concatenate([c[1] for c in chunks])
        if ndvi_all.size > args.max_pixels_per_class:
            idx = rng.choice(ndvi_all.size, size=args.max_pixels_per_class, replace=False)
            ndvi_all, rep_all = ndvi_all[idx], rep_all[idx]
        class_data[cls] = (ndvi_all, rep_all)
        print(f"  [{cls}] total kept pixels: {ndvi_all.size:,}")

    if not class_data:
        print("no usable cubes — nothing to save")
        return

    # KDE grid (matches the original 2D / 3D plot ranges: NDVI 0..1, REP 650..800)
    ndvi_grid = np.linspace(0.0, 1.0, args.grid_n_ndvi)
    rep_grid = np.linspace(650.0, 800.0, args.grid_n_rep)
    kde_df = kde_grid_csv(class_data, ndvi_grid, rep_grid)
    kde_out = out_dir / "ndvi_rep_kde_grid.csv"
    kde_df.to_csv(kde_out, index=False)
    print(f"  saved KDE grid     -> {kde_out}  rows={len(kde_df):,}")

    # Summary stats
    summ = summary_csv(class_data)
    summ_out = out_dir / "ndvi_rep_summary.csv"
    summ.to_csv(summ_out, index=False)
    print(f"  saved per-class summary -> {summ_out}")

    # Optional raw pixel table
    if args.save_pixels and pixel_rows:
        pix_out = out_dir / "ndvi_rep_pixels.csv"
        pd.concat(pixel_rows, ignore_index=True).to_csv(pix_out, index=False)
        print(f"  saved per-pixel table -> {pix_out}")


if __name__ == "__main__":
    main()
