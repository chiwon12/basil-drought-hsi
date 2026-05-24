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

---

## 1. Repository layout

```
git_open/
├── preprocessing/                                   # Python source — pipeline stages (flat layout)
│   ├── load_utils.py                                # cube I/O + reflectance helper
│   ├── vegetation_index.py                          # NDVI / NDWI / GNDVI / PRI
│   ├── reflectance_correction.py                    # raw → reflectance (white/dark)
│   ├── mask_swir_two_stage.py                       # SWIR leaf mask (raw thr → reflectance thr)
│   ├── mask_vnir_ndvi.py                            # VNIR leaf mask (NDVI + largest CC)
│   ├── make_train_test_split.py                     # plant-stratified 80/20 split
│   └── make_per_cluster_csvs.py                     # per-cluster mean-spectra CSVs
├── analysis/
│   └── compute_ndvi_rep_kde.py                      # per-class NDVI×REP KDE grid → CSV
├── normalization/
│   └── spectral_normalization.py                    # SNV / MSC functions
├── clustering/
│   └── pca_kmeans_clustering.py                     # PCA + K-Means (K=3) leaf clustering
├── augmentation/
│   ├── augment_with_autoencoder.py                  # AE-based latent interpolation
│   └── augment_with_gan.py                          # per-class 1D GAN sampling
├── models/
│   ├── classical_ml_models.py                       # RF / SVM / XGB / KNN + Optuna
│   └── cnn_1d_model.py                              # 1D-CNN + Optuna K-Fold
├── training/
│   ├── run_single_experiment.py                     # one (sensor,norm,data,model) combo
│   └── run_all_experiments.py                       # sweep all combinations
├── data/                                            # Small CSV / PNG artifacts (tracked in git, 13 MB)
│   ├── train_test_split/
│   │   ├── train_swir.csv   train_vnir.csv          # leaf-mean spectra + label (train)
│   │   └── test_swir.csv    test_vnir.csv           # leaf-mean spectra + label (test)
│   ├── augmentation/
│   │   ├── ae_swir_n300.csv     ae_vnir_n300.csv    # AE-interpolated (300/class)
│   │   └── gan_swir_n300.csv    gan_vnir_n300.csv   # GAN-sampled (300/class)
│   └── clustering/
│       ├── cluster_ratios_swir.csv                  # per-cube cluster pixel ratios (SWIR)
│       ├── cluster_ratios_vnir.csv                  # per-cube cluster pixel ratios (VNIR)
│       └── sample_images/                           # 8 PNG previews
│           ├── SW_WW_1-1_20250421_raw_swir_cube.png        #  K=3 cluster map, well-watered SWIR
│           ├── SW_MD_2-1_20250418_raw_swir_cube.png        #  mild drought
│           ├── SW_DR_2-1_20250421_raw_swir_cube.png        #  drought
│           ├── SW_SD_3-1_20250421_raw_swir_cube.png        #  severe drought
│           └── VN_{WW,MD,DR,SD}_*_raw_vnir_cube.png        #  same four treatments, VNIR
│
├── zenodo_data/                                     # Large binaries — hosted on Zenodo (NOT in git, ~40 GB)
│   ├── wavelengths/
│   │   ├── wavelengths_swir.npy                     #   (140,) band centers in nm
│   │   └── wavelengths_vnir.npy                     #   (120,) band centers in nm
│   ├── references/                                  # ── White / dark reference cubes (npy, float32)
│   │   ├── swir/{SW_dark_reference_E6, SW_white_reference_E6}_raw_swir_cube.npy        #  175 MB each
│   │   └── vnir/{VN_{dark,white}_reference_E20, VN_{dark,white}_reference_E30}_raw_vnir_cube.npy #  E20 (4/15+) + E30 (4/14)
│   ├── envi/                                        # ── Original ENVI .hdr/.raw (25 plates per sensor)
│   │   ├── swir/<run_dir>/SW_BS_<plate>_E6_<date>_raw{,.hdr}     # 25 plates × (.hdr + .raw)
│   │   └── vnir/<run_dir>/VN_BS_<plate>_E{20,30}_<date>_raw{,.hdr} # 25 plates × (.hdr + .raw)
│   ├── cubes/                                       # ── Per-plant cut cubes (final ML input, 100/sensor)
│   │   ├── swir/SW_{WW,MD,DR,SD}_{X-Y}_{date}_raw_swir_cube.npy  # 100 cubes, (256,320,140) float32, 44 MB each
│   │   └── vnir/VN_{WW,MD,DR,SD}_{X-Y}_{date}_raw_vnir_cube.npy  # 100 cubes, (512,640,120) float32, 151 MB each
│   │                                                #   Balanced subset (N=4 plants/slot):
│   │                                                #     WW (plate 1-1_1-4): 4 plants × 10 days = 40 cubes
│   │                                                #     MD (plate 2-1_2-4, 04/14-18) + DR (plate 2-1_2-4, 04/19-23) = 20+20 = 40
│   │                                                #     SD (plate 3-1_3-4, 04/19-23): 4 × 5 = 20 cubes
│   └── masks/                                       # ── Per-plant binary leaf masks (uint8, 100/sensor)
│       ├── swir/SW_{WW,MD,DR,SD}_{X-Y}_{date}_raw_swir_cube_mask.npy    # 100 files (256,320) uint8
│       └── vnir/VN_{WW,MD,DR,SD}_{X-Y}_{date}_raw_vnir_cube_mask.npy    # 100 files (512,640) uint8
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
The dataset goes through 4 processing stages — the Zenodo bundle ships
samples from every stage so users can enter the pipeline at any point.

| Bundle dir                 | Format          | Spatial (SWIR / VNIR) | Per-file size | Files (per sensor) |
|----------------------------|------------------|------------------------|---------------|---------------------|
| `envi/{swir,vnir}/`        | ENVI .hdr + .raw | 512×640 / 1024×1280     | ~175 / ~600 MB | **25** plate dirs    |
| `references/{swir,vnir}/`  | .npy float32     | 512×640 / 1024×1280     | ~175 / ~600 MB | 2 (SWIR) / 4 (VNIR)  |
| `cubes/{swir,vnir}/`       | .npy float32     | **256×320 / 512×640**   | ~44 / ~151 MB  | **100** cubes        |
| `masks/{swir,vnir}/`       | .npy uint8       | 256×320 / 512×640        | KB             | **100** masks        |
| `wavelengths/`             | .npy float32     | (140,) / (120,)          | ~1 KB          | 2                    |

- Bands: SWIR = 140 channels @ 900–1700 nm; VNIR = 120 channels @ 400–1000 nm.
- All cubes are `float32`, saved with `numpy.save`. The original ENVI files are
  shipped so reviewers can reproduce the pipeline from raw sensor counts.
- The per-plant cubes in `cubes/` are 1/4 the spatial area of the plate
  cubes (one plate of 4 plants → 4 cut cubes, one per plant).
- Reflectance cubes, plate-level NPYs, and intermediate stages are **not**
  shipped — they are deterministically reproducible from `envi/` via the
  preprocessing scripts.

### 2.3 Masks
- Binary leaf masks (uint8, 0=background, 1=leaf).
- SWIR mask shape (256, 320) aligns with the SW per-plant cut cube.
- VNIR mask shape (512, 640) aligns with the VN per-plant cut cube.

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
| ENVI (.hdr/.raw) → .npy conversion     | `preprocessing/envi_to_npy.py` |
| Reflectance correction (raw → R)       | `preprocessing/reflectance_correction.py` |
| SWIR leaf mask (two-stage thresholds)  | `preprocessing/mask_swir_two_stage.py` |
| VNIR leaf mask (NDVI + largest CC)     | `preprocessing/mask_vnir_ndvi.py` |
| Per-class NDVI × REP KDE → CSV         | `analysis/compute_ndvi_rep_kde.py` |
| Train/test split (plant-stratified)    | `preprocessing/make_train_test_split.py` |
| Spectral normalization (SNV / MSC)     | `normalization/spectral_normalization.py` |
| PCA + K-Means (K=3) leaf clustering    | `clustering/pca_kmeans_clustering.py` |
| AE-based spectral augmentation         | `augmentation/augment_with_autoencoder.py` |
| GAN-based spectral augmentation        | `augmentation/augment_with_gan.py` |
| Classical ML (RF / SVM / XGB / KNN)    | `models/classical_ml_models.py` |
| 1D-CNN                                 | `models/cnn_1d_model.py` |
| Single experiment driver               | `training/run_single_experiment.py` |
| Full sweep                             | `training/run_all_experiments.py` |

All scripts resolve paths relative to the repository root using
`Path(__file__).resolve().parents[2]`, so no editing is required after a
`git clone`.

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
python preprocessing/envi_to_npy.py --sensor sw
python preprocessing/envi_to_npy.py --sensor vn

# 2b. Raw → reflectance (uses references/{sensor}/)
python preprocessing/reflectance_correction.py --sensor sw
python preprocessing/reflectance_correction.py --sensor vn

# 2c. SWIR two-stage masking (consumes raw_cubes/swir/, computes reflectance internally)
python preprocessing/mask_swir_two_stage.py

# 2d. VNIR NDVI masking (consumes reflectance_cubes/vnir/)
python preprocessing/mask_vnir_ndvi.py --wavelengths zenodo_data/wavelengths/wavelengths_vnir.npy

# 3. Per-class NDVI × REP KDE distribution (CSV, no plots) — uses the bundled per-plant cubes
python analysis/compute_ndvi_rep_kde.py

# 4. Run one ML experiment (SWIR + SNV + raw + KNN) — uses data/train_test_split/
python training/run_single_experiment.py \
    --sensor sw --norm snv --data raw --model knn --mode all_leaf
```

### 3.4 Sensor codes used in CLI flags
`--sensor sw` ↔ SWIR cubes/CSVs (`*_swir.*`)
`--sensor vn` ↔ VNIR cubes/CSVs (`*_vnir.*`)

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
zenodo_data/cubes/{swir,vnir}/  +  zenodo_data/masks/{swir,vnir}/   ← shipped: 100 cubes + 100 masks/sensor
        │
        ▼  modeling (clustering, augmentation, classification)
data/train_test_split/  data/augmentation/  data/clustering/    ← shipped CSVs/PNGs (in git)
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
