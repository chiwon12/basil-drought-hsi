# Data Description

All paths are **relative to the repository root**. Code lives
under `src/`; data is split between the in-repo `data/` tree (small, tracked in
git) and the `zenodo_data/` bundle (large binaries, hosted on Zenodo).

## 1. Dataset overview

Basil (*Ocimum basilicum*) leaves imaged daily over 10 consecutive days
(2025-04-14 → 2025-04-23) under four irrigation treatments, with two
hyperspectral cameras (SWIR + VNIR).

| Item | Value |
|------|-------|
| Imaging days | 10 |
| Treatments | 4 (WW, MD, DR, SD) |
| SWIR cubes (full dataset) | 600 |
| VNIR cubes (full dataset) | 588 |
| Bundled subset (this release) | **100 per sensor** (N=4 plants/slot) |

Full-dataset per-treatment counts (for reference; the bundled subset is in §4):

| Treatment | SWIR | VNIR |
|-----------|------|------|
| WW | 200 | 196 |
| MD | 200 | 193 |
| DR | 100 | 101 |
| SD | 100 | 96 |

## 2. Where each artifact lives, and what uses it

### Zenodo bundle — `zenodo_data/` (~40 GB, large binaries)

| Path | Content | Shape (SWIR / VNIR) | Count/sensor | Used by |
|------|---------|----------------------|--------------|---------|
| `envi/{swir,vnir}/<run>/` | ENVI originals (`*.hdr`+`*.raw`, plate-level, uncorrected counts) | 512×640 / 1024×1280 | 25 plate dirs | `src/preprocessing/envi_to_npy.py` (stage 0) |
| `cubes/{swir,vnir}/` | **Per-plant cut cubes** (reflectance, final ML input) | 256×320×140 / 512×640×120 | 100 | clustering, masking, NDVI, feature extraction |
| `references/{swir,vnir}/` | White / dark reference cubes (`.npy` float32) | 512×640 / 1024×1280 | 2 SWIR / 4 VNIR | `src/preprocessing/reflectance_correction.py`, `mask_swir_two_stage.py` |

Intermediate stages (plate-level reflectance, per-plate mask outputs) are **not**
shipped — they regenerate from `envi/` via the preprocessing scripts.

### In-repo data — `data/` (~234 MB, tracked in git)

| Path | Content | Used by |
|------|---------|---------|
| `train_test_split/` | plant-aware split CSVs (§3.1) + reports | `run_single_experiment.py`, augmentation cache build |
| `per_cluster_split/` | per-cluster spectra + `c{0,1,2}/` plant-aware splits (§3.2) | cluster-mode `run_single_experiment.py --splits_dir` |
| `augmentation/cache_{swir,vnir}/` | AE/GAN fold caches (§3.3) | `run_single_experiment.py --data ae\|gan` |
| `wavelengths/wavelengths_{swir,vnir}.npy` | band centers in nm, (140,)/(120,) | masking, NDVI/REP analysis |
| `before_leaky_sw.csv` | naive-split (leaky) baseline results | `src/analysis/build_tables.py` (before/after table) |
| `cube_products/{swir,vnir}/` | per-cube derived products (§3.4) | reference / produced by masking + clustering |

## 3. File specifications

### 3.0 Hyperspectral cubes
- **SWIR** plate `(512,640,140)` → per-plant cut `(256,320,140)`, 900–1700 nm.
- **VNIR** plate `(1024,1280,120)` → per-plant cut `(512,640,120)`, 400–1000 nm.
- ENVI plates store uncorrected counts; `cubes/` hold reflectance in `[0,1]`
  after `R = (raw-dark)/(white-dark)`. `.npy` float32, load with `np.load`.

```
# plate-level (plant range)
SW_BS_1-1_1-4_20250421_raw_swir_cube.npy        # covers plants 1-1..1-4
# per-plant (label-renamed) — cubes/, masks/, cluster_*/ all use this stem
SW_WW_1-1_20250421_raw_swir_cube.npy
```

### 3.1 Train/test split CSV  (`data/train_test_split/`)
One row = one leaf-level mean spectrum. Files per sensor:
`train_<s>.csv`, `test_<s>.csv`, `full_pool_<s>.csv`, `manifest_<s>.csv`
(+ `split_report_{sw,vn}.json`).

| Column | Description |
|--------|-------------|
| `sample_id` | pivot sample id |
| `cluster_sample_id` | full cube stem |
| `plant` | physical plant key `<label>_<tray-plant>` (the CV group) |
| `date` | imaging date |
| `label` | treatment class (WW/MD/DR/SD) |
| `split` | `train` / `test` |
| `fold` | StratifiedGroupKFold index 0–4 (`-1` for test rows) |
| `feat_{nm}` | reflectance at band `nm` (one column per band) |

> **Split rule:** plant-level 80:20 (class-stratified) for the
> hold-out, and **StratifiedGroupKFold(5) grouped by `plant`** for CV — the same
> plant never appears in both train and validation/test. `split_report_*.json`
> records per-class plant/image counts and confirms zero plant leakage.

### 3.2 Per-cluster splits  (`data/per_cluster_split/`)
- `c{0,1,2}_{train,test}_{swir,vnir}.csv` — source per-cluster mean spectra.
- `c{0,1,2}/{train,test,full_pool,manifest}_{swir,vnir}.csv` — those spectra
  re-assigned to the **same plant-aware split/fold** as the all-leaf data
  (produced by `src/preprocessing/prepare_cluster_splits.py`; same columns as §3.1).

### 3.3 Augmentation caches  (`data/augmentation/cache_{swir,vnir}/`)
The **exact AE/GAN caches used in the paper are shipped here** (also regenerable
with the augmentation scripts' `--build_cache`). Built **inside each CV fold** to
avoid leakage. Target = 300 samples/class.

| File | Built from | Used when |
|------|------------|-----------|
| `{ae,gan}_fold{0..4}.npz` | that fold's train (`fold != k`) | CV fold k (training part) |
| `{ae,gan}_full.npz` | the whole new-train | final model (→ hold-out test) |

Each `.npz` holds `X` (`feat_*`, float32) and `y` (int labels), originals
included. `--data raw` ignores these.

### 3.4 Per-cube products  (`data/cube_products/{swir,vnir}/`)
All matched 1:1 to the cubes by filename stem, 100 per sensor.

| Subfolder / file | Content |
|------------------|---------|
| `masks/*_mask.npy` | binary leaf mask (`uint8`, 0=bg, 1=leaf), cube spatial shape |
| `mask_previews/*.png` | PNG preview of each mask |
| `cluster_labels/*_refined.npy` | K=3 cluster map (`uint8`: 0/1/2 clusters, 255=bg) |
| `cluster_previews/*_refined_blackbg.png` | color preview of each cluster map |
| `cluster_model/` | the fitted clustering pipeline (below) |
| `cluster_ratios.csv` | per-cube cluster pixel ratios: `sample, class, cluster_0, cluster_1, cluster_2` (cluster_0 = largest) |

`cluster_model/` (the single K=3 model that produced every label map, so new
cubes can be clustered identically without refitting):
`scaler_mean.npy`/`scaler_scale.npy` (band standardization), `pca_components.npy`
(40×120) + `pca_mean.npy` (95% variance), `kmeans_cluster_centers.npy` (3×40,
centers in PCA space), `pca_explained_variance_ratio.npy`,
`refined_cluster_ratio_summary.csv`.

### 3.5 Reference cubes  (`zenodo_data/references/{swir,vnir}/`)
White (diffuse panel) and dark (shutter closed) reference cubes, the inputs to
`R = (raw-dark)/(white-dark)` in `reflectance_correction.py`.

## 4. Bundled subset coverage (100 per sensor)

Balanced **N=4 plants per slot**, one plate per (treatment × period):

| Treatment | Plate | Plants | Days (n) | Cubes/sensor |
|-----------|-------|--------|----------|--------------|
| WW | `1-1_1-4` | 1-1, 1-2, 1-3, 1-4 | 2025-04-14 → 04-23 (10) | 40 |
| MD | `2-1_2-4` | 2-1, 2-2, 2-3, 2-4 | 2025-04-14 → 04-18 (5) | 20 |
| DR | `2-1_2-4` | 2-1, 2-2, 2-3, 2-4 | 2025-04-19 → 04-23 (5) | 20 |
| SD | `3-1_3-4` | 3-1, 3-2, 3-3, 3-4 | 2025-04-19 → 04-23 (5) | 20 |
| **Total** | 3 plates | 12 plants | 25 plate-days | **100** |

ENVI plates are shipped for every imaging day of these 3 plates (25
acquisitions/sensor), enabling end-to-end re-running of preprocessing.
Reproducing the full plant population (200 WW / 200 MD / 100 DR / 100 SD per
sensor) requires the complete ~921 GB dataset, available from the corresponding
author on reasonable request. See `data_mapping.md` for per-plant ↔ plate ↔
quadrant correspondence.
