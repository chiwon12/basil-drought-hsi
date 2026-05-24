"""
White / dark reference reflectance correction (raw → reflectance).

For each raw cube the corresponding white and dark reference cubes are matched
by a substring in the filename (e.g. acquisition date). The formula is:

    R = (raw - dark) / (white - dark)

clipped to [0, 1].

Default I/O layout (relative to git_open/):
    raw      : zenodo_data/raw_cubes/{sensor}/*.npy
    refs     : zenodo_data/references/{sensor}/*.npy
                  (dark cubes contain "dark" in their filename, white cubes "white")
    output   : zenodo_data/reflectance_cubes/{sensor}/*.npy

Use the CLI flags to override any of these.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def build_ref_map(ref_dir: Path) -> tuple[dict[str, Path], Path]:
    """Scan ref_dir for white_*.npy and dark_*.npy and build:
        - white_map: {key_substring -> white_npy_path}
        - dark_path: the dark reference (single file or first match)
    Filenames are expected to contain "white" or "dark"; the key_substring
    used for matching white refs is the 8-digit acquisition date if present,
    otherwise the file stem.
    """
    whites = sorted(p for p in ref_dir.glob("*.npy") if "white" in p.name.lower())
    darks = sorted(p for p in ref_dir.glob("*.npy") if "dark" in p.name.lower())
    if not darks:
        raise FileNotFoundError(f"no dark reference *.npy in {ref_dir}")
    if not whites:
        raise FileNotFoundError(f"no white reference *.npy in {ref_dir}")

    import re
    white_map: dict[str, Path] = {}
    for p in whites:
        m = re.search(r"(\d{8})", p.name)
        key = m.group(1) if m else p.stem
        white_map[key] = p
    return white_map, darks[0]


def correct_one(raw_path: Path, white: np.ndarray, dark: np.ndarray,
                out_path: Path) -> None:
    raw = np.load(raw_path)
    refl = (raw - dark) / (white - dark + 1e-8)
    refl = np.clip(refl, 0.0, 1.0).astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, refl)
    print(f"  saved {out_path.name}  shape={refl.shape}  range=[{refl.min():.3f},{refl.max():.3f}]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor", choices=["sw", "vn"], required=True,
                    help="swir or vnir")
    ap.add_argument("--raw_dir", default=None,
                    help="directory of raw *.npy cubes (default: zenodo_data/raw_cubes/{sensor})")
    ap.add_argument("--ref_dir", default=None,
                    help="directory of white/dark reference *.npy "
                         "(default: zenodo_data/references/{sensor})")
    ap.add_argument("--out_dir", default=None,
                    help="output directory (default: zenodo_data/reflectance_cubes/{sensor})")
    args = ap.parse_args()

    sensor_long = {"sw": "swir", "vn": "vnir"}[args.sensor]
    raw_dir = Path(args.raw_dir) if args.raw_dir else REPO / "zenodo_data" / "raw_cubes" / sensor_long
    ref_dir = Path(args.ref_dir) if args.ref_dir else REPO / "zenodo_data" / "references" / sensor_long
    out_dir = Path(args.out_dir) if args.out_dir else REPO / "zenodo_data" / "reflectance_cubes" / sensor_long

    print(f"[reflectance] sensor={args.sensor}")
    print(f"  raw_dir : {raw_dir}")
    print(f"  ref_dir : {ref_dir}")
    print(f"  out_dir : {out_dir}")

    white_map, dark_path = build_ref_map(ref_dir)
    dark = np.load(dark_path)
    print(f"  dark    : {dark_path.name}  shape={dark.shape}")
    print(f"  whites  : {len(white_map)} cube(s) ({list(white_map.keys())})")

    raws = sorted(raw_dir.glob("*.npy"))
    print(f"  raws    : {len(raws)} cube(s)")
    for raw_p in raws:
        # match white by date substring; else fall back to first white
        key = next((k for k in white_map if k in raw_p.name), next(iter(white_map)))
        white = np.load(white_map[key])
        out_p = out_dir / raw_p.name
        correct_one(raw_p, white, dark, out_p)


if __name__ == "__main__":
    main()
