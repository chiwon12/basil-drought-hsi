# Code Description

All paths in this document are **relative to `git_open/`** (the repository root).
Every script resolves data paths via `Path(__file__).resolve().parents[2]`, so
no edits are needed after cloning the repository.

## preprocessing/
| File | Purpose |
|------|---------|
| `load_utils.py` | I/O helpers (`load_cube`, `load_wavelengths`, `load_envi_cube`) and the canonical per-pixel reflectance correction formula `(raw - dark) / (white - dark)` clipped to [0, 1]. Imported by the masking scripts. |
| `vegetation_index.py` | Vegetation indices used by the manuscript: `compute_ndvi`, `compute_ndwi`, `compute_gndvi`, `compute_pri` plus a `find_band_index` helper. |
| `envi_to_npy.py` | **Stage 0 — ENVI → NPY.** Loads each `*.hdr/*.raw` under `zenodo_data/envi/{sensor}/<run_dir>/` via the `spectral` package and saves a float32 `.npy` cube under `zenodo_data/raw_cubes/{sensor}/`. Strips the `_E6/_E20/_E30_` exposure marker so the output filename matches the rest of the pipeline. |
| `reflectance_correction.py` | **Stage 1 — Raw → Reflectance.** Scans a directory of raw cube `.npy` files, matches each one to a white/dark reference (by date substring in the filename), applies `(raw - dark) / (white - dark)`, clips to [0, 1], and saves the result. Defaults: `zenodo_data/raw_cubes/{sensor}/` → `zenodo_data/reflectance_cubes/{sensor}/`. |
| `mask_swir_two_stage.py` | **Stage 2 — SWIR leaf masking (manuscript algorithm).** Two-stage thresholding: **(a)** on the RAW cube `R_1300 ≥ 6.8` → coarse leaf-vs-soil mask, **(b)** apply reflectance correction internally, then `R_1500 ≥ 0.26` → background mask; final = (a) AND NOT(b), with `remove_small_objects` / `remove_small_holes` / `closing(disk(3))` cleanup at both stages. Reads RAW cubes + white/dark refs directly (no separate reflectance step needed), writes binary `*_final_leaf_mask.npy`, a PNG preview, a whole-plate mean spectrum CSV, and four per-quadrant mean spectrum CSVs to `zenodo_data/mask_outputs/swir/`. |
| `mask_vnir_ndvi.py` | **Stage 2 — VNIR leaf masking.** Savitzky-Golay spectral smoothing → NDVI = `(R_870 - R_650) / (R_870 + R_650)` → median filter → manual threshold (default 0.12) → closing → keep largest connected component. Output: binary `uint8` masks under `zenodo_data/mask_outputs/vnir/`. |
| `make_train_test_split.py` | Stratified train/test split by **plant `sample_id`** (prevents temporal leakage from the same plant appearing on both sides). Pass `--pivot_dir <path>` to point at your local `pivot_labeled_{sw,vn}_data.csv` files. Output: `data/train_test_split/{train,test}_{swir,vnir}.csv`. |
| `make_per_cluster_csvs.py` | Re-builds per-cluster mean-spectrum CSVs (modes `all_leaf`, `c0`, `c1`, `c2`) from cube + K-Means label NPYs. Requires environment variables `CUBE_DIR_SW`, `CUBE_DIR_VN`, `LABEL_DIR_SW`, `LABEL_DIR_VN`, `WL_SW`, `WL_VN`. Output: `data/per_cluster_split/`. |

## normalization/
| File | Purpose |
|------|---------|
| `spectral_normalization.py` | Library of spectral normalizations: `apply_snv` (Standard Normal Variate), `apply_msc` (Multiplicative Scatter Correction, on-the-fly). Imported by `training/run_single_experiment.py`. |

## clustering/
| File | Purpose |
|------|---------|
| `pca_kmeans_clustering.py` | PCA (95 % variance) → K-Means (K=3) on leaf pixels. Cluster ids are remapped so 0 = largest cluster. CLI: `--sensor {sw,vn} --mask_source {file,ndvi,otsu_cube,self_nonzero}`. Requires `CUBE_DIR_*`, `MASK_DIR_*`, `WL_*` env vars (and optionally `OTSU_CUBE_DIR_VN`). Output: `data/cluster_outputs_k{K}/{sw,vn}/`. |

## augmentation/
| File | Purpose |
|------|---------|
| `augment_with_autoencoder.py` | Train a 1D autoencoder on leaf-mean spectra and synthesize new spectra by latent interpolation within each class. CLI: `--sensor {sw,vn} --target 300`. Default input/output paths point at `data/train_test_split/` and `data/augmentation/`. |
| `augment_with_gan.py` | Train a small per-class 1D GAN and sample synthetic spectra. Same CLI shape as above. |

## models/
| File | Purpose |
|------|---------|
| `classical_ml_models.py` | RandomForest, SVM, XGBoost, KNN with Optuna hyperparameter tuning + 5-fold CV. Reads `data/train_test_split/` (scope=all_leaf) or `data/per_cluster_split/` (scope=c0/c1/c2). Use `--aug_root` to override the augmentation CSV directory (default: `data/augmentation/`). |
| `cnn_1d_model.py` | 1D-CNN classifier (TensorFlow/Keras) with Optuna K-Fold CV and EarlyStopping. Pick sensor via `SENSOR=sw` (default) or `SENSOR=vn` environment variable. |

## training/
| File | Purpose |
|------|---------|
| `run_single_experiment.py` | Single-experiment driver: load split → normalize → (optional augmentation) → fit model → log metrics. The "experiment unit" used throughout the paper. |
| `run_all_experiments.py` | Sweeps `run_single_experiment.py` over all 5 normalizations × 6 models × 2 sensors and writes a results matrix CSV. Honors the `PYTHON` env var if your interpreter is not on `PATH`. |

## analysis/
| File | Purpose |
|------|---------|
| `compute_ndvi_rep_kde.py` | For every VNIR reflectance cube, compute NDVI and the red-edge position (REP = argmax of the spectral gradient in 680–750 nm), keep leaf pixels (NDVI > 0.3), group by treatment class (extracted from the filename), and save **CSV only — no plots**: a per-class Gaussian-KDE density grid on the 0–1 × 650–800 nm plane, per-class summary statistics, and (with `--save_pixels`) the underlying per-pixel table. Default input: `zenodo_data/cubes/*vnir*.npy`; default output: `results/ndvi_rep_kde/`. |

## Typical pipeline order

```
0. envi_to_npy.py                   # ENVI .hdr/.raw  -> raw cube .npy (per-plate)
1. reflectance_correction.py        # raw cube -> reflectance (white / dark refs)
                                    #   (skipped for SWIR if you use the
                                    #    two-stage masker, which corrects internally)
1b. mask_swir_two_stage.py          # SWIR leaf mask (raw threshold + reflectance threshold)
    mask_vnir_ndvi.py               # VNIR leaf mask (NDVI + largest connected component)
2. (per-plant cube cut)             # plate -> 4 per-plant cubes (paper Sec. 2.x)
3. make_train_test_split.py         # train/test split  (already provided)
4. pca_kmeans_clustering.py         # per-pixel K=3 cluster maps
5. make_per_cluster_csvs.py         # per-cluster mean-spectrum CSVs (mode c0/c1/c2/all_leaf)
6. augment_with_autoencoder.py      # (optional) AE-based spectral augmentation
   augment_with_gan.py              # (optional) GAN-based spectral augmentation
7. classical_ml_models.py           # ML sweep with Optuna
   cnn_1d_model.py                  # 1D-CNN sweep
   training/run_all_experiments.py  # full grid (CNN only)
8. analysis/compute_ndvi_rep_kde.py # per-class NDVI x REP KDE CSV
```
