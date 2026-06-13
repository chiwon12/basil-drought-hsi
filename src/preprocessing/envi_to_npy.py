"""
ENVI (.hdr/.raw) → .npy converter (Stage 0 of the preprocessing pipeline).

The Zenodo bundle ships the original ENVI files for 25 plates per sensor under
`zenodo_data/envi/{swir,vnir}/<run_dir>/<name>_raw.{hdr,raw}`. This script
loads each .hdr (via the `spectral` package), converts it to float32, and
writes the cube as `<name>_raw_{swir,vnir}_cube.npy` under
`zenodo_data/raw_cubes/{sensor}/`. The downstream scripts
(`reflectance_correction.py`, `mask_swir_two_stage.py`) consume that directory.

Usage
-----
    python preprocessing/envi_to_npy.py --sensor sw
    python preprocessing/envi_to_npy.py --sensor vn

The exposure marker (E6 / E20 / E30) is stripped from the output stem so the
filenames match the rest of the pipeline (e.g.
`SW_BS_1-1_1-4_E6_20250421_raw.hdr` → `SW_BS_1-1_1-4_20250421_raw_swir_cube.npy`).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
_EXP_RE = re.compile(r"_E(?:6|20|30)_")


def strip_exposure(stem: str) -> str:
    return _EXP_RE.sub("_", stem)


def convert_one(hdr_path: Path, out_path: Path) -> None:
    from spectral import open_image
    img = open_image(str(hdr_path))
    cube = np.asarray(img.load()).astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, cube)
    print(f"  {hdr_path.parent.name}/{hdr_path.name}  ->  {out_path.name}  shape={cube.shape}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor", choices=["sw", "vn"], required=True)
    ap.add_argument("--envi_root", default=None,
                    help="ENVI root (default: zenodo_data/envi/{sensor})")
    ap.add_argument("--out_dir", default=None,
                    help="output dir (default: zenodo_data/raw_cubes/{sensor})")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    sensor_long = {"sw": "swir", "vn": "vnir"}[args.sensor]
    sensor_tag = "swir_cube" if args.sensor == "sw" else "vnir_cube"
    envi_root = Path(args.envi_root) if args.envi_root else REPO / "zenodo_data" / "envi" / sensor_long
    out_dir = Path(args.out_dir) if args.out_dir else REPO / "zenodo_data" / "raw_cubes" / sensor_long

    hdrs = sorted(envi_root.rglob("*.hdr"))
    if not hdrs:
        raise FileNotFoundError(f"no *.hdr under {envi_root}")
    print(f"[envi->npy] sensor={args.sensor}  {len(hdrs)} hdr files")
    print(f"  envi_root: {envi_root}")
    print(f"  out_dir  : {out_dir}")

    for hdr in hdrs:
        stem = strip_exposure(hdr.stem)
        out_path = out_dir / f"{stem}_{sensor_tag}.npy"
        if out_path.exists() and not args.overwrite:
            print(f"  [skip] {out_path.name}")
            continue
        convert_one(hdr, out_path)
    print(f"\nDone -> {out_dir}")


if __name__ == "__main__":
    main()
