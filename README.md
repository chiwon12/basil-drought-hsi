# Basil Drought Hyperspectral Classification — Code & Sample Data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20368943.svg)](https://doi.org/10.5281/zenodo.20368943)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository accompanies the manuscript
**"[Manuscript title to be filled]"**
submitted to *[Journal name]*.

It provides:
- the analysis pipeline used to produce all figures and tables, and
- a representative sample of the hyperspectral imaging (HSI) data so that the
  pipeline can be executed end-to-end on a small subset.

The full raw dataset (~921 GB of hyperspectral cubes) is available from the
corresponding author upon reasonable request.

> **Leakage-safe evaluation.** The train/test split is performed at the **plant**
> level (not on individual leaf spectra), and 5-fold cross-validation uses
> **StratifiedGroupKFold** grouped by plant, so the same physical plant can never
> appear in both train and validation/test. Spectral normalization and data
> augmentation are fit **inside each fold** (training rows only). See
> [§3.6 Split & evaluation protocol](#36-split--evaluation-protocol).

---

## 1. Repository layout

Top-level folders:

| Folder | What it holds |
|--------|----------------|
| `src/`  | All Python source — the analysis pipeline (preprocessing → masking → clustering → augmentation → modeling → analysis). |
| `data/` | Small derived data **tracked in git** (~223 MB): plant-aware split CSVs, per-cube masks & cluster maps, wavelengths, augmentation caches. |
| `docs/` | Extended documentation — data, code, and per-plant↔plate mapping descriptions. |
| `zenodo_data/` | Large raw/ML binaries **hosted on Zenodo, not in git** (~40 GB): ENVI originals, per-plant cubes, reference cubes. |

```
basil-drought-hsi/
├── src/                                             # All Python source (pipeline stages)
│   ├── preprocessing/
│   │   ├── load_utils.py                            # cube I/O + reflectance helper
│   │   ├── vegetation_index.py                      # NDVI / NDWI / GNDVI / PRI
│   │   ├── envi_to_npy.py                           # ENVI .hdr/.raw → raw cube .npy
│   │   ├── reflectance_correction.py                # raw → reflectance (white/dark)
│   │   ├── mask_swir_two_stage.py                   # SWIR leaf mask (raw thr → reflectance thr)
│   │   ├── mask_vnir_ndvi.py                        # VNIR leaf mask (NDVI + largest CC)
│   │   ├── make_train_test_split.py                 # plant-level 80/20 + StratifiedGroupKFold(5)
│   │   ├── make_per_cluster_csvs.py                 # per-cluster mean-spectra CSVs
│   │   └── prepare_cluster_splits.py                # re-assign clusters to the plant-aware split
│   ├── analysis/compute_ndvi_rep_kde.py             # per-class NDVI×REP KDE grid → CSV
│   ├── normalization/spectral_normalization.py      # SNV / MSC (+ leakage-safe normalize_train_test)
│   ├── clustering/pca_kmeans_clustering.py          # PCA + K-Means (K=3) leaf clustering
│   ├── augmentation/
│   │   ├── augment_with_autoencoder.py              # AE latent interpolation (--build_cache, in-fold)
│   │   └── augment_with_gan.py                      # per-class 1D GAN (--build_cache, in-fold)
│   ├── models/
│   │   ├── classical_ml_models.py                   # RF / SVM / XGB / KNN builders + Optuna spaces
│   │   └── cnn_1d_model.py                          # PaperCNN 1D-CNN (PyTorch, paper 1d_cnn.py) + CV fit
│   └── training/
│       ├── run_single_experiment.py                 # one (sensor,norm,data,model) combo
│       └── run_all_experiments.py                   # sweep all combinations
├── data/                                            # Tracked in git (~223 MB): CSVs + small derived binaries
│   ├── train_test_split/                            # plant-aware split (meta cols + fold + feat_*)
│   │   ├── train_{swir,vnir}.csv  test_{swir,vnir}.csv     # split=train/test rows
│   │   ├── full_pool_{swir,vnir}.csv  manifest_{swir,vnir}.csv  # pooled + sample_id→split/fold map
│   │   └── split_report_{sw,vn}.json                # plant/fold counts + leakage check (=0)
│   ├── per_cluster_split/                           # per-cluster spectra + c{0,1,2}/ plant-aware splits
│   ├── augmentation/                                # cache_{swir,vnir}/{ae,gan}_*.npz (rebuilt, gitignored)
│   ├── wavelengths/wavelengths_{swir,vnir}.npy      # (140,) / (120,) band centers in nm
│   ├── before_leaky_sw.csv                          # naive-split (leaky) baseline for the before/after table
│   └── cube_products/                               # ALL per-cube derived products, grouped by sensor
│       └── {swir,vnir}/                             #   (100 cubes/sensor, matched 1:1 by filename)
│           ├── masks/                               #     binary leaf masks (uint8 .npy)
│           ├── mask_previews/                       #     mask preview PNGs
│           ├── cluster_labels/                      #     K=3 cluster maps (uint8 .npy; 0/1/2, 255=bg)
│           ├── cluster_previews/                    #     cluster preview PNGs
│           ├── cluster_model/                       #     fitted scaler + PCA + KMeans + ratio summary
│           └── cluster_ratios.csv                   #     per-cube cluster pixel ratios
│
├── zenodo_data/                                     # Large raw/ML binaries — hosted on Zenodo (NOT in git, ~40 GB)
│   ├── envi/                                        # ── Original ENVI .hdr/.raw (25 plates per sensor)
│   │   ├── swir/<run_dir>/SW_BS_<plate>_E6_<date>_raw{,.hdr}     # 25 plates × (.hdr + .raw)
│   │   └── vnir/<run_dir>/VN_BS_<plate>_E{20,30}_<date>_raw{,.hdr} # 25 plates × (.hdr + .raw)
│   ├── cubes/                                       # ── Per-plant cut cubes (final ML input, 100/sensor)
│   │   ├── swir/SW_{WW,MD,DR,SD}_{X-Y}_{date}_raw_swir_cube.npy  # 100 cubes, (256,320,140) float32
│   │   └── vnir/VN_{WW,MD,DR,SD}_{X-Y}_{date}_raw_vnir_cube.npy  # 100 cubes, (512,640,120) float32
│   │                                                #   Balanced subset (N=4 plants/slot): WW 40, MD 20, DR 20, SD 20
│   └── references/                                  # ── White / dark reference cubes (npy, float32)
│       ├── swir/{SW_dark_reference_E6, SW_white_reference_E6}_raw_swir_cube.npy
│       └── vnir/VN_{dark,white}_reference_E{20,30}_raw_vnir_cube.npy   # E20 (4/15+) + E30 (4/14)
│
├── docs/                                            # Extended documentation
│   ├── data_description.md
│   ├── code_description.md
│   └── data_mapping.md                              # ← per-plant ↔ plate ↔ quadrant mapping
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

---

## 2. Data description

### 2.1 Experimental design
- **Crop**: Basil (*Ocimum basilicum*)
- **Treatments** (4 classes):
  - `WW` — Well-Watered (control)
  - `MD` — Mild Drought
  - `DR` — Drought (moderate)
  - `SD` — Severe Drought
- **Imaging period**: 2025-04-14 to 2025-04-23 (10 consecutive days)

### 2.2 Hyperspectral cubes
The large binaries live on **Zenodo** (`zenodo_data/`, ~40 GB); the small derived
products ship **in this repo** under `data/`.

| Location | Dir | Format | Spatial (SWIR / VNIR) | Files (per sensor) |
|----------|-----|--------|------------------------|---------------------|
| Zenodo | `envi/{swir,vnir}/`        | ENVI .hdr + .raw | 512×640 / 1024×1280   | **25** plate dirs   |
| Zenodo | `cubes/{swir,vnir}/`       | .npy float32     | **256×320 / 512×640** | **100** cubes       |
| Zenodo | `references/{swir,vnir}/`  | .npy float32     | 512×640 / 1024×1280   | 2 (SWIR) / 4 (VNIR) |
| repo `data/` | `cube_products/{swir,vnir}/masks/`          | .npy uint8  | 256×320 / 512×640 | **100** masks      |
| repo `data/` | `cube_products/{swir,vnir}/mask_previews/`  | .png        | —                 | **100** previews   |
| repo `data/` | `cube_products/{swir,vnir}/cluster_labels/` | .npy uint8  | 256×320 / 512×640 | **100** label maps |
| repo `data/` | `wavelengths/`             | .npy float32 | (140,) / (120,)      | 2                   |

- Bands: SWIR = 140 channels @ 900–1700 nm; VNIR = 120 channels @ 400–1000 nm.
- All cubes are `float32`, saved with `numpy.save`. The original ENVI files are
  shipped so reviewers can reproduce the pipeline from raw sensor counts.
- The per-plant cubes in `cubes/` are 1/4 the spatial area of the plate
  cubes (one plate of 4 plants → 4 cut cubes, one per plant).
- Reflectance cubes, plate-level NPYs, and intermediate stages are **not**
  shipped — they are deterministically reproducible from `envi/` via the
  preprocessing scripts.

### 2.3 Masks
- Binary leaf masks (uint8, 0=background, 1=leaf), **100 per sensor, 1:1 with the
  cubes** (every `*_cube.npy` has a `*_cube_mask.npy`).
- SWIR mask shape (256, 320) aligns with the SW per-plant cut cube.
- VNIR mask shape (512, 640) aligns with the VN per-plant cut cube.
- `data/cube_products/{swir,vnir}/mask_previews/` provides a PNG preview per mask (100/sensor).
- **NDVI** is a VNIR-only vegetation index, not shipped as a separate product: it
  is computed on demand from the VNIR cubes via `src/preprocessing/mask_vnir_ndvi.py`
  (binary NDVI leaf mask) and `src/analysis/compute_ndvi_rep_kde.py` (per-class
  NDVI×REP distribution), so it is fully reproducible from the cubes.

### 2.3b Per-cube clustering (`data/cube_products/{swir,vnir}/`)
K=3 PCA + K-Means leaf clustering, matched 1:1 to the cubes:
- `cluster_labels/*_refined.npy` — per-pixel cluster map (uint8: 0/1/2 leaf
  clusters, 255 = background), same spatial shape as the cube.
- `cluster_previews/*_refined_blackbg.png` — color preview.
- `cluster_model/` — the fitted pipeline (`scaler_*`, `pca_*`,
  `kmeans_cluster_centers.npy`) + `refined_cluster_ratio_summary.csv`;
  `cluster_ratios.csv` holds the per-cube cluster pixel ratios.
- Coverage: **SWIR 100/100** and **VNIR 100/100** — every cube has a matching
  label map + preview. New clustering can be regenerated for any cube with
  `src/clustering/pca_kmeans_clustering.py`.

### 2.4 Representative samples (Zenodo, ~40 GB)

**Balanced N=4 subset** — one plate per (treatment × day) slot, 4 plants/plate:

| Slot                         | Plate     | Days                              | Plants                  | Cubes (per sensor) |
|------------------------------|-----------|------------------------------------|--------------------------|---------------------|
| WW (Well-Watered)            | `1-1_1-4` | 2025-04-14 → 04-23 (10 d)         | 1-1, 1-2, 1-3, 1-4       | 40                  |
| MD (early)                   | `2-1_2-4` | 2025-04-14 → 04-18 (5 d)          | 2-1, 2-2, 2-3, 2-4       | 20                  |
| DR (late)                    | `2-1_2-4` | 2025-04-19 → 04-23 (5 d)          | 2-1, 2-2, 2-3, 2-4       | 20                  |
| SD (Severe Drought)          | `3-1_3-4` | 2025-04-19 → 04-23 (5 d)          | 3-1, 3-2, 3-3, 3-4       | 20                  |
| **Total**                    | 3 plates  | 25 plate-days                      | 12 plants                | **100**             |

ENVI plates are shipped for **every** imaging day of these 3 plates (25 plate
acquisitions per sensor), enabling end-to-end re-running of preprocessing.

---

## 3. Reproducing the manuscript

### 3.1 Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Tested with Python 3.10 on Ubuntu 22.04.

### 3.2 Main entry points (mapped to manuscript sections)

| Manuscript topic                       | Script |
|----------------------------------------|--------|
| ENVI (.hdr/.raw) → .npy conversion     | `src/preprocessing/envi_to_npy.py` |
| Reflectance correction (raw → R)       | `src/preprocessing/reflectance_correction.py` |
| SWIR leaf mask (two-stage thresholds)  | `src/preprocessing/mask_swir_two_stage.py` |
| VNIR leaf mask (NDVI + largest CC)     | `src/preprocessing/mask_vnir_ndvi.py` |
| Per-class NDVI × REP KDE → CSV         | `src/analysis/compute_ndvi_rep_kde.py` |
| Plant-level split + StratifiedGroupKFold(5) | `src/preprocessing/make_train_test_split.py` |
| Per-cluster plant-aware split          | `src/preprocessing/prepare_cluster_splits.py` |
| Spectral normalization (SNV / MSC, fit per fold) | `src/normalization/spectral_normalization.py` |
| PCA + K-Means (K=3) leaf clustering    | `src/clustering/pca_kmeans_clustering.py` |
| AE augmentation (in-fold caches)       | `src/augmentation/augment_with_autoencoder.py --build_cache` |
| GAN augmentation (in-fold caches)      | `src/augmentation/augment_with_gan.py --build_cache` |
| Classical ML (RF / SVM / XGB / KNN)    | `src/models/classical_ml_models.py` |
| 1D-CNN                                 | `src/models/cnn_1d_model.py` |
| Single experiment driver               | `src/training/run_single_experiment.py` |
| Full sweep                             | `src/training/run_all_experiments.py` |

All scripts resolve data paths relative to the repository root (via
`Path(__file__).resolve().parents[2]` from `src/<area>/<script>.py`), so no
editing is required after a `git clone`. The `src/training/` driver also adds
`src/normalization/` and `src/models/` to `sys.path` to import the shared
model/normalization helpers.

### 3.3 Quick demo on the representative subset

After downloading the Zenodo bundle and extracting it into `zenodo_data/`:

```bash
# 1. Verify cube shapes
python - <<'PY'
import numpy as np
c = np.load("zenodo_data/cubes/swir/SW_WW_1-1_20250421_raw_swir_cube.npy")
print(c.shape, c.dtype)   # -> (256, 320, 140) float32
PY

# 2a. ENVI (.hdr/.raw) → .npy raw cubes (writes zenodo_data/raw_cubes/{sensor}/)
python src/preprocessing/envi_to_npy.py --sensor sw
python src/preprocessing/envi_to_npy.py --sensor vn

# 2b. Raw → reflectance (uses zenodo_data/references/{sensor}/)
python src/preprocessing/reflectance_correction.py --sensor sw
python src/preprocessing/reflectance_correction.py --sensor vn

# 2c. SWIR two-stage masking (consumes raw_cubes/swir/, computes reflectance internally)
python src/preprocessing/mask_swir_two_stage.py

# 2d. VNIR NDVI masking (consumes reflectance_cubes/vnir/)
python src/preprocessing/mask_vnir_ndvi.py --wavelengths data/wavelengths/wavelengths_vnir.npy

# 3. Per-class NDVI × REP KDE distribution (CSV, no plots) — uses the bundled per-plant cubes
python src/analysis/compute_ndvi_rep_kde.py

# 4. (optional) build leakage-safe in-fold augmentation caches
python src/augmentation/augment_with_autoencoder.py --sensor sw --build_cache
python src/augmentation/augment_with_gan.py         --sensor sw --build_cache

# 5. Run one experiment (SWIR + SNV + raw + KNN) — uses data/train_test_split/
python src/training/run_single_experiment.py --sensor sw --norm snv --data raw --model knn
# ...or the full 3 norm × 3 data × 5 model sweep for a sensor:
python src/training/run_all_experiments.py --sensor sw
```

### 3.4 Sensor codes used in CLI flags
`--sensor sw` ↔ SWIR cubes/CSVs (`*_swir.*`)
`--sensor vn` ↔ VNIR cubes/CSVs (`*_vnir.*`)

### 3.6 Split & evaluation protocol

Because each plant is imaged on several dates, splitting on individual leaf
spectra would let rows of one plant fall on both sides of the split (and across
CV folds), inflating the cross-validated scores. The protocol below keeps every
plant on a single side:

| Aspect | Protocol |
|--------|----------|
| Split unit | **plant** — class-stratified 80/20 over plants |
| Cross-validation | **StratifiedGroupKFold(5)** grouped by plant |
| Normalization (SNV/MSC + StandardScaler) | **fit per fold** on the fold's train part only |
| Augmentation (AE/GAN) | **per-fold caches** built from `fold != k` rows; full-train cache for the final fit |
| Final test | plant-level holdout, **always raw**, never used to pick params/epochs |
| Selection metric | mean **weighted-F1** across folds |
| CNN final epochs | fixed to the **mean best-epoch measured during CV** |

`data/train_test_split/split_report_{sw,vn}.json` records the per-class plant/
image counts and confirms zero train↔test (and fold↔fold) plant leakage.

### 3.5 End-to-end pipeline order

```
zenodo_data/envi/{swir,vnir}/<run>/<name>_raw.hdr+.raw         ← Stage 0: ENVI originals (shipped)
        │  envi_to_npy.py
        ▼
zenodo_data/raw_cubes/{sensor}/<name>_raw_<sensor>_cube.npy    ← Stage 1: plate cubes, pre-reflectance (regenerated)
        │  reflectance_correction.py  (uses references/{sensor}/)
        ▼
zenodo_data/reflectance_cubes/{sensor}/                        ← Stage 2: plate cubes, reflectance
        │  mask_swir_two_stage.py  /  mask_vnir_ndvi.py
        ▼
zenodo_data/mask_outputs/{sensor}/                             ← per-plate leaf masks (regenerated)
        │  (per-plant cut — 1/4 spatial area)
        ▼
zenodo_data/cubes/{swir,vnir}/                                 ← Zenodo: 100 cut cubes/sensor
data/cube_products/{swir,vnir}/{masks,cluster_labels,...}/     ← in git: per-cube masks + cluster maps
        │
        ▼  modeling (clustering, augmentation, classification)
data/train_test_split/  data/augmentation/  data/cube_products/ ← shipped CSVs/PNGs/small npy (in git)
```

---

## 4. Licensing
- **Code**: MIT License — see `LICENSE`.
- **Data (Zenodo)**: CC BY 4.0.

## 5. Citation
If you use this code or data, please cite:
```
[BibTeX entry to be filled after acceptance]
```
A DOI for the data bundle is issued by Zenodo at release time:
**Zenodo DOI**: [10.5281/zenodo.20368943](https://doi.org/10.5281/zenodo.20368943)

## 6. Contact
For questions or to request the full raw dataset:
**[Corresponding author]** — `[email]`
