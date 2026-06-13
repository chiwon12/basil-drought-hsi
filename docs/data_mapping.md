# Per-plant ↔ Plate Mapping Reference

The Zenodo bundle ships per-plant cut cubes (`zenodo_data/cubes/`) and the
original plate-level ENVI files (`zenodo_data/envi/`). This document shows how
each per-plant filename maps back to its source plate and quadrant inside the
plate.

A "plate" is the field of view captured in one hyperspectral scan and contains
**4 basil plants arranged in a 2 × 2 grid** (top-left / top-right / bottom-left /
bottom-right quadrant). The per-plant cubes in `cubes/` are obtained by spatial
slicing of the corresponding reflectance-corrected plate cube.

---

## 1. Filename conventions

### ENVI plate originals (`envi/{swir,vnir}/<run_dir>/`)
```
{SW|VN}_BS_{a}-{b}_{a}-{c}_E{6|20|30}_{nnn}_{YYYYMMDD}_{HHMMSS}/
    {SW|VN}_BS_{a}-{b}_{a}-{c}_E{6|20|30}_{YYYYMMDD}_raw          (binary ENVI .raw)
    {SW|VN}_BS_{a}-{b}_{a}-{c}_E{6|20|30}_{YYYYMMDD}_raw.hdr      (ENVI header)
```

The exposure marker is `E6` for SWIR, `E20` for VNIR (04/15+), and `E30` for
VNIR on 04/14.

### Per-plant cut cubes (`cubes/{swir,vnir}/`) and masks (`masks/{swir,vnir}/`)
```
{SW|VN}_{WW|MD|DR|SD}_{tray}-{plant}_{YYYYMMDD}_raw_{swir|vnir}_cube.npy
{SW|VN}_{WW|MD|DR|SD}_{tray}-{plant}_{YYYYMMDD}_raw_{swir|vnir}_cube_mask.npy

example: SW_WW_1-1_20250421_raw_swir_cube.npy
         VN_DR_2-1_20250421_raw_vnir_cube_mask.npy
```

Field meanings:
| Field | Values |
|-------|--------|
| `{SW|VN}` | sensor — SWIR or VNIR |
| `{WW|MD|DR|SD}` | irrigation treatment class |
| `{tray}` | 1, 2, 3 |
| `{plant}` | 1, 2, 3, 4 (position number within the tray) |
| `{YYYYMMDD}` | imaging date |

---

## 2. Plate → 4 plants (quadrant mapping)

A plate cube of shape `(H, W, B)` is split into four per-plant cubes of shape
`(H/2, W/2, B)`:

| Quadrant      | Spatial slice (numpy)         | Plant offset | Example (plate `1-1_1-4`) |
|---------------|--------------------------------|--------------|---------------------------|
| top-left      | `cube[0:H/2,   0:W/2,   :]`   | 0            | plant **1-1**             |
| top-right     | `cube[0:H/2,   W/2:W,   :]`   | 1            | plant **1-2**             |
| bottom-left   | `cube[H/2:H,   0:W/2,   :]`   | 2            | plant **1-3**             |
| bottom-right  | `cube[H/2:H,   W/2:W,   :]`   | 3            | plant **1-4**             |

For SWIR plates `(512, 640, 140)` → each per-plant `(256, 320, 140)`.
For VNIR plates `(1024, 1280, 120)` → each per-plant `(512, 640, 120)`.

### Reverse lookup: given a per-plant filename, find its plate and quadrant
1. Plate start index: `start = ((plant - 1) // 4) * 4 + 1`
2. Plate end index:   `end = start + 3`
3. Plate name:        `{SW|VN}_BS_{tray}-{start}_{tray}-{end}_{date}_raw_{swir|vnir}_cube`
4. Quadrant offset:   `quad = (plant - start)` → 0=top_left, 1=top_right, 2=bottom_left, 3=bottom_right

Example: `SW_DR_2-1_20250421_raw_swir_cube.npy` →
plate `SW_BS_2-1_2-4_20250421_raw_swir_cube` → quadrant `top_left`.

---

## 3. Bundled subset (N=4 per slot, 100 cubes per sensor)

| Treatment | Plants            | Plate (origin) | Imaging days        | Cubes/sensor |
|-----------|-------------------|----------------|---------------------|--------------|
| WW        | 1-1, 1-2, 1-3, 1-4 | `1-1_1-4`     | 10 (04-14 → 04-23)  | 40           |
| MD        | 2-1, 2-2, 2-3, 2-4 | `2-1_2-4`     | 5  (04-14 → 04-18)  | 20           |
| DR        | 2-1, 2-2, 2-3, 2-4 | `2-1_2-4`     | 5  (04-19 → 04-23)  | 20           |
| SD        | 3-1, 3-2, 3-3, 3-4 | `3-1_3-4`     | 5  (04-19 → 04-23)  | 20           |
| **Total** | 12 plants          | 3 plates       | 25 plate-days       | **100**      |

ENVI plates are shipped for every imaging day of these 3 plates (25 plate
acquisitions per sensor), enabling end-to-end pipeline re-runs.

MD and DR draw from the same physical plate (`2-1_2-4`) on different dates —
treatment labels change across the experimental schedule (Section 2 of the
manuscript).

---

## 4. Cube ↔ mask correspondence

For every per-plant cube there is one matching binary mask:

| Per-plant cube                                    | Binary mask                                           | Shape           |
|---------------------------------------------------|-------------------------------------------------------|-----------------|
| `cubes/swir/SW_*_raw_swir_cube.npy`               | `masks/swir/SW_*_raw_swir_cube_mask.npy`              | (256, 320) uint8 |
| `cubes/vnir/VN_*_raw_vnir_cube.npy`               | `masks/vnir/VN_*_raw_vnir_cube_mask.npy`              | (512, 640) uint8 |

The cube and mask filenames are identical aside from the `_mask` suffix, so a
1:1 lookup is `mask_path = cube_path.replace("/cubes/", "/masks/").replace(".npy", "_mask.npy")`.

> **Note on early-date VNIR ENVI filenames**: VNIR plates on 04-14 use the
> `E30` exposure marker and 04-15 onwards use `E20`. The per-plant filenames
> in `cubes/vnir/` strip the exposure marker for uniformity — no special
> handling needed on the consumer side.
