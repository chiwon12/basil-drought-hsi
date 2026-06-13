# -*- coding: utf-8 -*-
"""
Leaf-only K-means clustering on hyperspectral cubes (SW or VN).

Pipeline (per sensor):
  1) Collect leaf pixels from all cubes -> training set
       --mask_source file : load binary mask npy (uint8, 0=bg, 1=leaf)
       --mask_source ndvi : compute NDVI from cube and Otsu-threshold per cube
                            (VN only; needs visible Red ~670 nm and NIR ~800 nm)
  2) Train: StandardScaler -> IncrementalPCA -> MiniBatchKMeans(k=K)
  3) Cluster ids are remapped so id 0 = most pixels, 1 = next, 2 = next
     -> palette becomes red / blue / green for the top-3 clusters.
  4) Dense-predict cluster label for every leaf pixel of every cube.
     Background pixels stay at 255 (rendered black).
  5) Save: refined NPY/PNG per cube + scaler/pca/kmeans params + ratio CSV.

Inputs (paths are set with environment variables — cubes/masks are not bundled):
  CUBE_DIR_SW / CUBE_DIR_VN : directories of *_cube.npy
       SW shape (256, 320, 140), VN shape (256, 320, 120)
  MASK_DIR_SW / MASK_DIR_VN : directories of *_cube_mask.npy (binary uint8, 0=bg, 1=leaf)
  WL_SW / WL_VN             : 1D numpy arrays of band centers (nm)
  OTSU_CUBE_DIR_VN          : (optional) pre-masked VN cubes for --mask_source otsu_cube

Output:
  data/cluster_outputs_k{K}/{sw|vn}/   (under git_open/)

Usage:
  python pca_kmeans_clustering.py --sensor sw                       # SW with mask file
  python pca_kmeans_clustering.py --sensor vn --mask_source ndvi    # VN with NDVI auto-mask
"""

import argparse
import os
import re
from glob import glob
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / 'data'


def _env(name: str, required: bool = True):
    v = os.environ.get(name)
    if required and not v:
        raise RuntimeError(f'environment variable {name} is required')
    return v


CUBE_DIR = {'sw': _env('CUBE_DIR_SW'), 'vn': _env('CUBE_DIR_VN')}
MASK_DIR = {'sw': _env('MASK_DIR_SW'), 'vn': _env('MASK_DIR_VN')}
WL_FILE = {'sw': _env('WL_SW'), 'vn': _env('WL_VN')}
OTSU_CUBE_DIR = {'vn': _env('OTSU_CUBE_DIR_VN', required=False)}

# Palette ordered so the largest cluster (rank 0 after remap) is red, next blue, next green.
PALETTE = [
    (255,   0,   0),  # 0 red    (most pixels)
    (  0,   0, 255),  # 1 blue   (second)
    (  0, 255,   0),  # 2 green  (third)
    (255, 255,   0),  # 3 yellow
    (255,   0, 255),  # 4 magenta
    (  0, 255, 255),  # 5 cyan
]


def cube_stem_to_mask_stem(stem: str) -> str:
    """Some VN cubes have '__' or 'corr_vnir' while masks use '_' and 'raw_vnir'."""
    s = re.sub(r'_+', '_', stem)
    s = s.replace('_corr_vnir', '_raw_vnir')
    return s


def find_mask_path(cube_path: str, mask_dir: str) -> str:
    stem = os.path.basename(cube_path).replace('_cube.npy', '')
    p = os.path.join(mask_dir, f"{stem}_cube_mask.npy")
    if os.path.exists(p):
        return p
    p = os.path.join(mask_dir, f"{cube_stem_to_mask_stem(stem)}_cube_mask.npy")
    return p if os.path.exists(p) else ''


def extract_class(name: str) -> str:
    for cls in ('WW', 'SD', 'DR', 'MD'):
        if f"_{cls}_" in name:
            return cls
    return "Unknown"


def otsu_threshold(values: np.ndarray, n_bins: int = 256, lo: float = -1, hi: float = 1) -> float:
    hist, edges = np.histogram(values, bins=n_bins, range=(lo, hi))
    p = hist / max(hist.sum(), 1)
    bins = (edges[:-1] + edges[1:]) / 2
    cum_p = np.cumsum(p)
    cum_pb = np.cumsum(p * bins)
    total_mean = cum_pb[-1]
    denom = cum_p * (1 - cum_p)
    sigma_b2 = np.where(denom > 0, (total_mean * cum_p - cum_pb) ** 2 / (denom + 1e-12), 0)
    return float(bins[int(np.argmax(sigma_b2))])


def make_ndvi_mask(cube: np.ndarray, wl: np.ndarray, fixed_threshold: float | None) -> tuple:
    """Compute NDVI = (NIR - Red)/(NIR + Red) and binarize via Otsu (or fixed)."""
    ir = int(np.argmin(np.abs(wl - 800)))
    rd = int(np.argmin(np.abs(wl - 670)))
    nir = cube[..., ir].astype(np.float32)
    red = cube[..., rd].astype(np.float32)
    ndvi = (nir - red) / (nir + red + 1e-6)
    thr = fixed_threshold if fixed_threshold is not None else otsu_threshold(ndvi)
    return (ndvi > thr).astype(np.uint8), thr


def get_leaf_mask(cube: np.ndarray, cube_path: str, sensor: str,
                  mask_source: str, wl: np.ndarray, fixed_thr: float | None):
    """Return (mask uint8 (H,W) matching cube spatial shape, info_dict)."""
    if mask_source == 'file':
        mp = find_mask_path(cube_path, MASK_DIR[sensor])
        if not mp:
            return None, {'reason': 'mask_missing'}
        m = np.load(mp)
        if m.shape != cube.shape[:2]:
            return None, {'reason': 'shape_mismatch'}
        return m.astype(np.uint8), {'mask_path': mp}
    if mask_source == 'ndvi':
        m, thr = make_ndvi_mask(cube, wl, fixed_thr)
        return m, {'threshold': thr}
    if mask_source == 'self_nonzero':
        # Cube is already mask-applied: any-band != 0 -> leaf; all-zero -> background.
        m = np.any(cube != 0, axis=2).astype(np.uint8)
        return m, {}
    if mask_source == 'otsu_cube':
        # Pre-masked full-res cube (any-band != 0). Downsample to current cube spatial shape.
        cube_name = os.path.basename(cube_path).replace('_cube.npy', '_cube_masked_cube.npy')
        cand = os.path.join(OTSU_CUBE_DIR[sensor], cube_name)
        if not os.path.exists(cand):
            # try normalized stem
            stem = os.path.basename(cube_path).replace('_cube.npy', '')
            cand = os.path.join(OTSU_CUBE_DIR[sensor],
                                f"{cube_stem_to_mask_stem(stem)}_cube_masked_cube.npy")
            if not os.path.exists(cand):
                return None, {'reason': 'otsu_cube_missing'}
        full = np.load(cand, mmap_mode='r')
        m_full = np.any(full != 0, axis=2)
        H, W = cube.shape[:2]
        Hf, Wf = m_full.shape
        if Hf == H and Wf == W:
            return m_full.astype(np.uint8), {'otsu_path': cand}
        if Hf % H == 0 and Wf % W == 0:
            ry, rx = Hf // H, Wf // W
            m_ds = m_full.reshape(H, ry, W, rx).max(axis=(1, 3))
            return m_ds.astype(np.uint8), {'otsu_path': cand, 'downsample': (ry, rx)}
        return None, {'reason': f'shape_mismatch otsu={m_full.shape} cube={cube.shape[:2]}'}
    raise ValueError(f'unknown mask_source: {mask_source}')


def collect_train_samples(sensor: str, max_per_file: int, max_total: int,
                          seed: int, mask_source: str, wl: np.ndarray,
                          fixed_thr: float | None):
    cube_paths = sorted(glob(os.path.join(CUBE_DIR[sensor], "*_cube.npy")))
    rng = np.random.default_rng(seed)
    X_list, taken, used, skipped = [], 0, 0, 0
    thresholds = []

    print(f"[1/3] {sensor.upper()}: collecting leaf-pixel training samples "
          f"(mask_source={mask_source}, cubes={len(cube_paths)}, "
          f"max_per_file={max_per_file:,}, max_total={max_total:,})")
    for p in cube_paths:
        if taken >= max_total:
            break
        cube = np.load(p, mmap_mode='r')
        mask, info = get_leaf_mask(cube, p, sensor, mask_source, wl, fixed_thr)
        if mask is None:
            skipped += 1
            continue
        if 'threshold' in info:
            thresholds.append(info['threshold'])

        idx = np.where(mask.ravel() == 1)[0]
        if idx.size == 0:
            skipped += 1
            continue
        if max_per_file and idx.size > max_per_file:
            idx = rng.choice(idx, size=max_per_file, replace=False)
        X_flat = cube.reshape(-1, cube.shape[2])
        X_list.append(np.asarray(X_flat[idx], dtype=np.float32))
        taken += idx.size
        used += 1

    if not X_list:
        raise RuntimeError(f"No training data for {sensor}")
    X = np.vstack(X_list)
    print(f"  -> used {used} cubes, skipped {skipped}, "
          f"total {X.shape[0]:,} pixels x {X.shape[1]} bands")
    if thresholds:
        t = np.array(thresholds)
        print(f"  NDVI threshold (Otsu) stats over {len(t)} cubes: "
              f"mean={t.mean():.3f}, median={np.median(t):.3f}, "
              f"min={t.min():.3f}, max={t.max():.3f}")
    return X


def train(X: np.ndarray, k: int, pca_dim: int, batch: int, seed: int, out_dir: str):
    print(f"[2/3] StandardScaler -> IncrementalPCA(d={pca_dim}) -> MiniBatchKMeans(k={k})")
    scaler = StandardScaler().fit(X)

    ipca = IncrementalPCA(n_components=pca_dim)
    for s in range(0, X.shape[0], batch):
        e = min(s + batch, X.shape[0])
        ipca.partial_fit((X[s:e] - scaler.mean_) / scaler.scale_)

    Xp = []
    for s in range(0, X.shape[0], batch):
        e = min(s + batch, X.shape[0])
        Xp.append(ipca.transform((X[s:e] - scaler.mean_) / scaler.scale_))
    Xp = np.vstack(Xp)

    km = MiniBatchKMeans(n_clusters=k, random_state=seed,
                         n_init=20, max_iter=200, batch_size=batch).fit(Xp)

    # --- Remap cluster ids: id 0 = most pixels, id 1 = next, ...
    counts = np.bincount(km.labels_, minlength=k)
    order = np.argsort(-counts)            # original_id list, descending by count
    inv = np.argsort(order)                # inv[orig] = new_rank
    km.cluster_centers_ = km.cluster_centers_[order]
    km.labels_ = inv[km.labels_]
    new_counts = counts[order]
    print("  cluster counts (after remap, id=0 most):",
          dict(zip(range(k), new_counts.tolist())))
    print("  PCA explained variance ratio sum:",
          float(ipca.explained_variance_ratio_.sum()))

    np.save(os.path.join(out_dir, "scaler_mean.npy"),  scaler.mean_)
    np.save(os.path.join(out_dir, "scaler_scale.npy"), scaler.scale_)
    np.save(os.path.join(out_dir, "pca_components.npy"), ipca.components_)
    np.save(os.path.join(out_dir, "pca_mean.npy"), ipca.mean_)
    np.save(os.path.join(out_dir, "pca_explained_variance_ratio.npy"),
            ipca.explained_variance_ratio_)
    np.save(os.path.join(out_dir, "kmeans_cluster_centers.npy"), km.cluster_centers_)
    return scaler, ipca, km


def dense_predict(sensor: str, scaler, ipca, km, k: int, out_dir: str,
                  pred_batch: int, mask_source: str, wl: np.ndarray,
                  fixed_thr: float | None):
    labels_dir = os.path.join(out_dir, "refined_labels_npy")
    png_dir = os.path.join(out_dir, "refined_png_blackbg")
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    rows = []
    cube_paths = sorted(glob(os.path.join(CUBE_DIR[sensor], "*_cube.npy")))
    print(f"[3/3] {sensor.upper()}: dense-predicting {len(cube_paths)} cubes")
    skipped = 0
    for i, p in enumerate(cube_paths, 1):
        sample = os.path.basename(p).replace("_cube.npy", "")
        cube = np.load(p, mmap_mode="r")
        mask, info = get_leaf_mask(cube, p, sensor, mask_source, wl, fixed_thr)
        if mask is None:
            skipped += 1
            continue
        H, W, C = cube.shape

        idx = np.where(mask.ravel() == 1)[0]
        if idx.size == 0:
            skipped += 1
            continue

        refined = np.full((H, W), 255, dtype=np.uint8)
        X_flat = cube.reshape(-1, C)
        preds = np.empty(idx.size, dtype=np.uint8)
        for s in range(0, idx.size, pred_batch):
            e = min(s + pred_batch, idx.size)
            Xb = (X_flat[idx[s:e]] - scaler.mean_) / scaler.scale_
            preds[s:e] = km.predict(ipca.transform(Xb))
        refined.ravel()[idx] = preds

        np.save(os.path.join(labels_dir, f"{sample}_refined.npy"), refined)

        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        for kc in range(k):
            rgb[refined == kc] = PALETTE[kc % len(PALETTE)]
        plt.imsave(os.path.join(png_dir, f"{sample}_refined_blackbg.png"), rgb)

        u, c = np.unique(preds, return_counts=True)
        total = int(c.sum()) if c.size else 1
        d = dict(zip(u.tolist(), c.tolist()))
        row = {"sample": sample, "class": extract_class(sample)}
        for kc in range(k):
            row[f"cluster_{kc}"] = d.get(kc, 0) / total * 100.0
        rows.append(row)

        if i % 50 == 0 or i == len(cube_paths):
            print(f"  [{i}/{len(cube_paths)}]")

    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "refined_cluster_ratio_summary.csv"), index=False)
    print(f"  saved labels -> {labels_dir}")
    print(f"  saved PNGs   -> {png_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sensor', choices=['sw', 'vn'], required=True)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--mask_source', choices=['file', 'ndvi', 'otsu_cube', 'self_nonzero'],
                        default='file',
                        help='file=binary mask npy; ndvi=auto from cube; '
                             'otsu_cube=derive from pre-masked cube in OTSU_CUBE_DIR; '
                             'self_nonzero=cube itself is masked (any-band!=0 -> leaf)')
    parser.add_argument('--ndvi_threshold', type=float, default=None,
                        help='fixed NDVI threshold; if None and mask_source=ndvi, use Otsu')
    parser.add_argument('--pca_dim', type=int, default=40)
    parser.add_argument('--train_batch', type=int, default=50_000)
    parser.add_argument('--pred_batch', type=int, default=100_000)
    parser.add_argument('--max_per_file', type=int, default=100_000)
    parser.add_argument('--max_total', type=int, default=2_000_000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    out_dir = str(DATA / f'cluster_outputs_k{args.k}' / args.sensor)
    os.makedirs(out_dir, exist_ok=True)
    print(f"=== Leaf-only K-means clustering | sensor={args.sensor} k={args.k} "
          f"mask_source={args.mask_source} ===")
    print(f"out_dir: {out_dir}")

    wl = np.load(WL_FILE[args.sensor])
    X = collect_train_samples(args.sensor, args.max_per_file, args.max_total, args.seed,
                              args.mask_source, wl, args.ndvi_threshold)
    scaler, ipca, km = train(X, args.k, args.pca_dim, args.train_batch, args.seed, out_dir)
    dense_predict(args.sensor, scaler, ipca, km, args.k, out_dir, args.pred_batch,
                  args.mask_source, wl, args.ndvi_threshold)
    print(f"\nDone -> {out_dir}")


if __name__ == "__main__":
    main()
