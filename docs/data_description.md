# Data Description

All paths in this document are **relative to `git_open/`** (the repository root).

## 1. Dataset overview

The full dataset consists of basil (*Ocimum basilicum*) leaves imaged daily
over 10 consecutive days (2025-04-14 to 2025-04-23) under four irrigation
treatments. Each plant was imaged with two hyperspectral cameras (SWIR + VNIR).

| Item                | Value                                       |
|---------------------|---------------------------------------------|
| Imaging days        | 10                                          |
| Treatments          | 4 (WW, MD, DR, SD)                          |
| SWIR cubes (total)  | 600                                         |
| VNIR cubes (total)  | 588                                         |
| SWIR mean-spectra rows | `data/train_test_split/{train,test}_swir.csv` |
| VNIR mean-spectra rows | `data/train_test_split/{train,test}_vnir.csv` |

### 1.1 Per-treatment counts (full dataset)
| Treatment | SWIR | VNIR |
|-----------|------|------|
| WW        | 200  | 196  |
| MD        | 200  | 193  |
| DR        | 100  | 101  |
| SD        | 100  |  96  |

## 2. File specifications

The bundled Zenodo subset is organized as the pipeline produces it, so you can
enter the workflow at any stage.

### 2.1 Zenodo bundle layout
| Folder | Content | Spatial shape (SWIR / VNIR) | Per-file size | Files (per sensor) |
|--------|---------|------------------------------|---------------|---------------------|
| `envi/{swir,vnir}/<run_dir>/` | ENVI originals (`*.hdr` + `*.raw`, plate-level, pre-reflectance) | 512×640 / 1024×1280 | ~175 / ~600 MB | **25** plate dirs |
| `references/{swir,vnir}/` | White/dark reference cubes (`.npy`, float32) | 512×640 / 1024×1280 | ~175 / ~600 MB | 2 SWIR / 4 VNIR (E20+E30) |
| `cubes/{swir,vnir}/` | **Cut per-plant cubes** (final ML input, reflectance-corrected) | **256×320 / 512×640** | **44 / 151 MB** | **100** cubes |
| `masks/{swir,vnir}/` | Per-plant binary leaf masks (`uint8`) | 256×320 / 512×640 | KB | **100** masks |
| `wavelengths/` | Band centers in nm | (140,) / (120,) | ~1 KB | 2 |

The intermediate stages (plate-level reflectance NPY, plate-level renamed
NPY, per-plate mask outputs) are **not shipped** — they are reproducible
from `envi/` via `preprocessing/envi_to_npy.py` + `reflectance_correction.py`
+ `mask_*.py`.

### 2.2 Hyperspectral cube  shapes
- **SWIR**
  - ENVI plate: `(H=512, W=640, C=140)`
  - Per-plant cut cube: `(H=256, W=320, C=140)`
  - Range 900–1700 nm.
- **VNIR**
  - ENVI plate: `(H=1024, W=1280, C=120)`
  - Per-plant cut cube: `(H=512, W=640, C=120)`
  - Range 400–1000 nm.
- ENVI plates store uncorrected sensor counts; cut cubes (in `cubes/`) hold
  reflectance values in `[0, 1]` after the per-pixel `R = (raw-dark)/(white-dark)`
  correction.
- NPY files are `float32`, saved with `numpy.save`. Load with `np.load(path)`.

Filename conventions:
```
# stage 1 / 2: per-plate (plant range)
{SW|VN}_BS_{a}-{b}_{a}-{c}_{YYYYMMDD}_raw_{swir|vnir}_cube.npy
SW_BS_1-1_1-4_20250421_raw_swir_cube.npy        # plate covers plants 1-1, 1-2, 1-3, 1-4

# stage 3 / 4b: per-plant (label-renamed)
{SW|VN}_{WW|MD|DR|SD}_{tray}-{plant}_{YYYYMMDD}_raw_{swir|vnir}_cube.npy
SW_WW_1-1_20250421_raw_swir_cube.npy
```

### 2.3 Reference cubes  (`zenodo_data/references/{swir,vnir}/`)
- **White reference**: same shape as a single raw cube, captured against a
  diffuse white reflectance panel (Spectralon-equivalent) under identical
  illumination/exposure.
- **Dark reference**: captured with the sensor shutter closed; represents the
  per-pixel dark current.
- These are the inputs to `preprocessing/reflectance_correction.py`:
  `R = (raw - dark) / (white - dark)`.

### 2.4 Leaf mask  (`zenodo_data/masks/{swir,vnir}/*_mask.npy`)
- SWIR: shape `(H=256, W=320)`, aligned 1:1 with `cubes/swir/`.
- VNIR: shape `(H=512, W=640)`, aligned 1:1 with `cubes/vnir/`.
- `uint8`, values `{0: background, 1: leaf}`.

> **Known issue (VNIR mask)**: the canonical `mask_vn_final_relabeled/`
> folder in the original dataset is corrupted (PNG-axis artifacts). For VNIR,
> use the NDVI-derived `*_masked_cube.npy` files (any band ≠ 0 → leaf) and
> regenerate the binary mask with `preprocessing/mask_vnir_ndvi.py`.

### 2.3 Train/test split CSV  (`data/train_test_split/*.csv`)
One row = one leaf-level mean spectrum.

| Column          | Description                                     |
|-----------------|-------------------------------------------------|
| `sample_id`     | `{tray}-{plant}` (consistent across days)       |
| `cluster_sample_id` | full cube stem, used to join with cluster ratios |
| `feat_{nm}`     | reflectance at wavelength `nm` (one column per band) |
| `label`         | treatment class (`WW`/`MD`/`DR`/`SD`)           |

> **Important — split rule**: train/test split is stratified by
> **plant `sample_id`**, not by image. The same plant imaged on different
> days stays on the same side of the split.

### 2.4 Augmented spectra  (`data/augmentation/*.csv`)
- Synthetic leaf-mean spectra produced by AE-interpolation or GAN sampling.
- Target = 300 samples per class → 1,200 rows per file.

| File                    | Method | Sensor |
|-------------------------|--------|--------|
| `ae_swir_n300.csv`      | AE     | SWIR   |
| `ae_vnir_n300.csv`      | AE     | VNIR   |
| `gan_swir_n300.csv`     | GAN    | SWIR   |
| `gan_vnir_n300.csv`     | GAN    | VNIR   |

Column layout: same `feat_*` + `label`, no `sample_id`.

### 2.5 Cluster ratios  (`data/clustering/cluster_ratios_{swir,vnir}.csv`)
For each cube, the per-cluster pixel ratio after PCA + K-Means refinement.

| Column            | Description                                  |
|-------------------|----------------------------------------------|
| `sample`          | cube filename stem                           |
| `class`           | treatment class (WW/MD/DR/SD)                |
| `cluster_0`       | % leaf pixels in cluster 0 (largest)         |
| `cluster_1`       | % in cluster 1                               |
| `cluster_2`       | % in cluster 2                               |

Cluster ids are remapped so `cluster_0` is always the largest by pixel count
(palette: red / blue / green).

### 2.6 Cluster sample previews  (`data/clustering/sample_images/`)
8 PNG previews of the refined K=3 cluster maps, one per
(treatment × sensor) combination. They are produced by
`clustering/pca_kmeans_clustering.py` and shown here as visual sanity
checks; they correspond to (or are temporally close to) the cubes bundled on
Zenodo.

| File                                       | Treatment | Sensor | Date       |
|--------------------------------------------|-----------|--------|------------|
| `SW_WW_1-1_20250421_raw_swir_cube.png`            | WW        | SWIR   | 2025-04-21 |
| `SW_MD_2-1_20250418_raw_swir_cube.png`            | MD        | SWIR   | 2025-04-18 |
| `SW_DR_2-1_20250421_raw_swir_cube.png`            | DR        | SWIR   | 2025-04-21 |
| `SW_SD_3-1_20250421_raw_swir_cube.png`            | SD        | SWIR   | 2025-04-21 |
| `VN_WW_1-1_20250421_raw_vnir_cube.png`            | WW        | VNIR   | 2025-04-21 |
| `VN_MD_2-1_20250418_raw_vnir_cube.png`            | MD        | VNIR   | 2025-04-18 |
| `VN_DR_2-1_20250421_raw_vnir_cube.png`            | DR        | VNIR   | 2025-04-21 |
| `VN_SD_3-1_20250421_raw_vnir_cube.png`            | SD        | VNIR   | 2025-04-21 |

> The MD samples are dated 2025-04-18 because MD plants were only clustered
> up to that date in the full pipeline; the other treatments are all
> matched to 2025-04-21 for consistency with the Zenodo cube bundle.

## 3. Sample coverage (stage 4b per-plant)

Two plants per treatment are bundled across **every imaging day** of that
treatment, both sensors:

| Treatment | Plants | Days covered (n) | Cubes per sensor |
|-----------|--------|-------------------|-------------------|
| WW | 1-1, 1-2 | 2025-04-14 → 2025-04-23 (10) | 20 |
| MD | 2-1, 2-2 | 2025-04-14 → 2025-04-18 (5)  | 10 |
| DR | 2-1, 2-2 | 2025-04-19 → 2025-04-23 (5)  | 10 |
| SD | 3-1, 3-2 | 2025-04-19 → 2025-04-23 (5)  | 10 |
| **Total** |  |  | **50 per sensor** |

This means the bundled subset is sufficient to:
- run PCA + K-Means clustering on real cubes
  (`clustering/pca_kmeans_clustering.py`) — at least 10 cubes per class,
- compute cluster pixel ratios and per-class statistics for the included plants,
  and
- compute the per-class NDVI × REP KDE distribution
  (`analysis/compute_ndvi_rep_kde.py`).

Reproducing the manuscript's full plant population (200 WW / 200 MD / 100 DR /
100 SD per sensor) requires the complete 921 GB dataset, which is available
from the corresponding author upon reasonable request.
