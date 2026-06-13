"""
SWIR leaf mask — two-stage thresholding pipeline used in the manuscript.

This script takes a RAW SWIR cube (uncorrected sensor counts) together with
white/dark reference cubes, applies the manuscript's two-stage masking
procedure, and saves the resulting binary leaf mask and per-quadrant mean
spectra.

Stage 1 — Leaf vs. soil  (on the RAW cube):
    leaf_raw = cube_raw[:, :, idx(1300 nm)]  >= leaf_raw_threshold (default 6.8)
  followed by remove_small_objects -> remove_small_holes -> closing(disk(r))
  -> remove_small_objects again.

Stage 2 — Leaf vs. background  (on the REFLECTANCE cube):
    The cube is reflectance-corrected via
        R = (raw - dark) / (white - dark)
    and a background mask is built from
        bg = R[:, :, idx(1500 nm)]  >= bg_corr_threshold (default 0.26)
    The final mask is (stage1) AND NOT(bg), then the same morphology cleanup
    is reapplied.

Outputs (under <out_dir>/<cube_stem>/):
    <stem>_final_leaf_mask.npy         binary uint8 (H, W)
    <stem>_final_leaf_mask.png         visualization
    <stem>_corr_avg.csv                whole-plate mean reflectance spectrum
    <stem>_corr_avg_part{1..4}.csv     mean spectrum for each plate quadrant
                                       (top_left / top_right / bottom_left / bottom_right)

Default I/O (relative to git_open/):
    raw     : zenodo_data/raw_cubes/swir/*.npy
    white   : zenodo_data/references/swir/*white*.npy
    dark    : zenodo_data/references/swir/*dark*.npy
    wl      : data/wavelengths/wavelengths_swir.npy
    output  : zenodo_data/mask_outputs/swir/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.morphology import closing, disk, remove_small_holes, remove_small_objects

from load_utils import apply_reflectance_correction, load_cube, load_wavelengths
from vegetation_index import find_band_index

REPO = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Two-stage segmentation                                                      #
# --------------------------------------------------------------------------- #
def segment_leaf_vs_soil_raw(cube_raw: np.ndarray,
                             wavelengths: np.ndarray,
                             leaf_nm_raw: float,
                             leaf_raw_threshold: float,
                             min_object_size: int = 200,
                             min_hole_size: int = 800,
                             closing_radius: int = 4,
                             island_size: int = 1500) -> np.ndarray:
    """Stage 1: high-reflectance leaf pixels on the RAW cube."""
    idx = find_band_index(wavelengths, leaf_nm_raw)
    band = cube_raw[:, :, idx].astype(np.float32)
    mask = band >= leaf_raw_threshold
    mask = remove_small_objects(mask, min_size=min_object_size)
    mask = remove_small_holes(mask, area_threshold=min_hole_size)
    mask = closing(mask, disk(closing_radius))
    mask = remove_small_objects(mask.astype(bool), min_size=island_size)
    return mask.astype(np.uint8)


def segment_leaf_vs_bg_reflectance(cube_corr: np.ndarray,
                                   wavelengths: np.ndarray,
                                   leaf_raw_mask: np.ndarray,
                                   bg_nm_corr: float,
                                   bg_corr_threshold: float,
                                   min_object_size: int = 200,
                                   min_hole_size: int = 800,
                                   closing_radius: int = 4,
                                   island_size: int = 1000) -> np.ndarray:
    """Stage 2: subtract reflectance-bright background pixels."""
    idx = find_band_index(wavelengths, bg_nm_corr)
    band = cube_corr[:, :, idx].astype(np.float32)
    mask_bg = band >= bg_corr_threshold
    leaf_final = leaf_raw_mask.astype(bool) & (~mask_bg)
    leaf_final = remove_small_objects(leaf_final, min_size=min_object_size)
    leaf_final = remove_small_holes(leaf_final, area_threshold=min_hole_size)
    leaf_final = closing(leaf_final, disk(closing_radius))
    leaf_final = remove_small_objects(leaf_final.astype(bool), min_size=island_size)
    return leaf_final.astype(np.uint8)


# --------------------------------------------------------------------------- #
# Per-cube driver                                                              #
# --------------------------------------------------------------------------- #
def process_cube(cube_path: Path,
                 white_cube: np.ndarray,
                 dark_cube: np.ndarray,
                 wavelengths: np.ndarray,
                 out_dir: Path,
                 leaf_nm_raw: float,
                 leaf_raw_threshold: float,
                 bg_nm_corr: float,
                 bg_corr_threshold: float,
                 min_object_size: int,
                 min_hole_size: int,
                 closing_radius: int,
                 island_size: int) -> None:
    print(f"\n▶ Processing: {cube_path.name}")
    cube_raw = load_cube(str(cube_path))
    H, W, B = cube_raw.shape

    # Stage 1: leaf vs soil on the raw cube
    mask_leaf_raw = segment_leaf_vs_soil_raw(
        cube_raw, wavelengths,
        leaf_nm_raw=leaf_nm_raw,
        leaf_raw_threshold=leaf_raw_threshold,
        min_object_size=min_object_size,
        min_hole_size=min_hole_size,
        closing_radius=closing_radius,
        island_size=island_size,
    )
    print(f"  [stage 1] leaf-raw pixels: {int(mask_leaf_raw.sum()):,} / {H*W:,}")

    # Reflectance correction
    cube_corr = apply_reflectance_correction(cube_raw, white_cube, dark_cube)
    band_bg = cube_corr[:, :, find_band_index(wavelengths, bg_nm_corr)].astype(np.float32)
    print(f"  [reflectance @{bg_nm_corr:.0f} nm] "
          f"min={band_bg.min():.3f}  median={np.median(band_bg):.3f}  max={band_bg.max():.3f}")

    # Stage 2: subtract bright-reflectance background from leaf candidates
    leaf_final_mask = segment_leaf_vs_bg_reflectance(
        cube_corr, wavelengths, mask_leaf_raw,
        bg_nm_corr=bg_nm_corr,
        bg_corr_threshold=bg_corr_threshold,
        min_object_size=min_object_size,
        min_hole_size=min_hole_size,
        closing_radius=closing_radius,
        island_size=island_size,
    )
    print(f"  [stage 2] final leaf pixels: {int(leaf_final_mask.sum()):,} / {H*W:,}")

    # ------------------- save outputs ----------------------------------- #
    stem = cube_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # binary mask (NPY) — primary downstream artifact
    np.save(out_dir / f"{stem}_final_leaf_mask.npy", leaf_final_mask)
    # PNG visualization
    plt.imsave(out_dir / f"{stem}_final_leaf_mask.png",
               leaf_final_mask * 255, cmap="gray")

    # whole-plate mean spectrum on the reflectance cube
    leaf_idxs = np.where(leaf_final_mask.ravel() == 1)[0]
    cube_corr_2d = cube_corr.reshape(-1, B)
    if leaf_idxs.size == 0:
        mean_corr_spec = np.zeros_like(wavelengths, dtype=np.float64)
    else:
        mean_corr_spec = cube_corr_2d[leaf_idxs].mean(axis=0)

    pd.DataFrame({"Wavelength (nm)": wavelengths,
                  "Corrected Mean Reflectance": mean_corr_spec}
                 ).to_csv(out_dir / f"{stem}_corr_avg.csv", index=False)

    # per-quadrant mean spectra (plate -> 4 plants)
    quad_names = ["top_left", "top_right", "bottom_left", "bottom_right"]
    quad_coords = [
        (slice(0, H // 2), slice(0, W // 2)),
        (slice(0, H // 2), slice(W // 2, W)),
        (slice(H // 2, H), slice(0, W // 2)),
        (slice(H // 2, H), slice(W // 2, W)),
    ]
    for i, ((hs, ws), qname) in enumerate(zip(quad_coords, quad_names), start=1):
        sub_mask = leaf_final_mask[hs, ws]
        sub_cube = cube_corr[hs, ws, :]
        if int(sub_mask.sum()) == 0:
            print(f"  ⚠ no leaf pixels in {qname}")
            mean_spec = np.zeros_like(wavelengths, dtype=np.float64)
        else:
            mean_spec = sub_cube[sub_mask.astype(bool)].mean(axis=0)

        col = f"Corrected Mean Reflectance (ROI-{qname})"
        pd.DataFrame({"Wavelength (nm)": wavelengths, col: mean_spec}
                     ).to_csv(out_dir / f"{stem}_corr_avg_part{i}.csv", index=False)
        print(f"  ▶ saved {qname:13s} spectrum -> {stem}_corr_avg_part{i}.csv")


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def _find_one(folder: Path, contains: str) -> Path:
    cand = sorted(p for p in folder.glob("*.npy") if contains in p.name.lower())
    if not cand:
        raise FileNotFoundError(f"no *.npy containing '{contains}' in {folder}")
    return cand[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default=None,
                    help="directory of RAW SWIR cube *.npy "
                         "(default: zenodo_data/raw_cubes/swir/)")
    ap.add_argument("--ref_dir", default=None,
                    help="directory containing white/dark reference cubes "
                         "(default: zenodo_data/references/swir/)")
    ap.add_argument("--out_dir", default=None,
                    help="output directory (default: zenodo_data/mask_outputs/swir/)")
    ap.add_argument("--wavelengths", default=None,
                    help="wavelengths .npy (default: data/wavelengths/wavelengths_swir.npy)")

    # Manuscript defaults
    ap.add_argument("--leaf_nm_raw",        type=float, default=1300.0)
    ap.add_argument("--leaf_raw_threshold", type=float, default=6.8)
    ap.add_argument("--bg_nm_corr",         type=float, default=1500.0)
    ap.add_argument("--bg_corr_threshold",  type=float, default=0.26)
    ap.add_argument("--min_object_size",    type=int,   default=300)
    ap.add_argument("--min_hole_size",      type=int,   default=500)
    ap.add_argument("--closing_radius",     type=int,   default=3)
    ap.add_argument("--island_size",        type=int,   default=5000)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir) if args.raw_dir else REPO / "zenodo_data" / "raw_cubes" / "swir"
    ref_dir = Path(args.ref_dir) if args.ref_dir else REPO / "zenodo_data" / "references" / "swir"
    out_dir = Path(args.out_dir) if args.out_dir else REPO / "zenodo_data" / "mask_outputs" / "swir"
    wl_path = Path(args.wavelengths) if args.wavelengths else REPO / "data" / "wavelengths" / "wavelengths_swir.npy"

    print("=== SWIR two-stage masking ===")
    print(f"  raw_dir : {raw_dir}")
    print(f"  ref_dir : {ref_dir}")
    print(f"  out_dir : {out_dir}")
    print(f"  wl      : {wl_path}")

    # Load references once (broadcast against every cube)
    white_path = _find_one(ref_dir, "white")
    dark_path = _find_one(ref_dir, "dark")
    print(f"  white   : {white_path.name}")
    print(f"  dark    : {dark_path.name}")
    white_cube = load_cube(str(white_path))
    dark_cube = load_cube(str(dark_path))
    wavelengths = load_wavelengths(str(wl_path))

    cubes = sorted(raw_dir.glob("*.npy"))
    print(f"  cubes   : {len(cubes)} file(s)")
    if not cubes:
        return

    for p in cubes:
        process_cube(
            cube_path=p,
            white_cube=white_cube,
            dark_cube=dark_cube,
            wavelengths=wavelengths,
            out_dir=out_dir,
            leaf_nm_raw=args.leaf_nm_raw,
            leaf_raw_threshold=args.leaf_raw_threshold,
            bg_nm_corr=args.bg_nm_corr,
            bg_corr_threshold=args.bg_corr_threshold,
            min_object_size=args.min_object_size,
            min_hole_size=args.min_hole_size,
            closing_radius=args.closing_radius,
            island_size=args.island_size,
        )


if __name__ == "__main__":
    main()
