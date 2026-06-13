# -*- coding: utf-8 -*-
"""AE-based spectral augmentation by latent interpolation (PyTorch).

Two changes vs the first release:
  1. PyTorch port of the original Keras autoencoder (identical architecture and
     hyper-parameters: enc Linear(64)->Linear(32), dec Linear(64)->Linear(D),
     MSE, 100 epochs, batch 64, Adam 1e-3; class-balanced 0.5/0.5 latent
     interpolation up to `target` per class).
  2. **Leakage-safe, fold-wise caches.** Augmentation must only ever see a
     fold's own training rows, so `--build_cache` reads the split CSV (which
     carries the `fold` column) and writes, per CV fold k, an augmentation built
     from rows with ``fold != k`` only, plus a `full` cache built from the whole
     new-train. Validation/test rows are never augmented.

`augment_ae(X_raw, y, target, seed)` is the importable library function used by
both this CLI and the CV loop in training/run_single_experiment.py.

Usage:
  python augment_with_autoencoder.py --sensor sw --build_cache   # per-fold + full npz
  python augment_with_autoencoder.py --sensor vn --build_cache
Output: data/augmentation/cache_{swir,vnir}/ae_fold{0..4}.npz, ae_full.npz
        (each holds X = raw-reflectance features, y = int labels; originals included)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / 'data'
SENSOR_LONG = {'sw': 'swir', 'vn': 'vnir'}
CLASS_NAMES = ['DR', 'MD', 'SD', 'WW']
N_FOLDS = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class _AE(nn.Module):
    def __init__(self, d, enc=32):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, enc), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(enc, 64), nn.ReLU(), nn.Linear(64, d))

    def forward(self, x):
        return self.decoder(self.encoder(x))


def augment_ae(X_raw, y, target=300, seed=42, epochs=100):
    """X_raw (n, D) reflectance, y (n,) int. Returns (X_out, y_out) including
    originals, each class padded to `target` by latent interpolation. All
    fitting (scaler + AE) happens on the rows passed in — never on val/test."""
    torch.manual_seed(seed); np.random.seed(seed)
    sc = StandardScaler().fit(X_raw)
    Xs = sc.transform(X_raw).astype(np.float32)
    ae = _AE(Xs.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    T = torch.from_numpy(Xs).to(DEVICE)
    n, bs = len(Xs), 64
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad(); crit(ae(T[idx]), T[idx]).backward(); opt.step()
    ae.eval()
    Xaug, yaug = [], []
    with torch.no_grad():
        for cls in np.unique(y):
            lat = ae.encoder(torch.from_numpy(Xs[y == cls]).to(DEVICE))
            n_gen = target - int((y == cls).sum())
            i = cnt = 0
            while cnt < n_gen and len(lat) >= 2:
                z = 0.5 * lat[i % len(lat)] + 0.5 * lat[(i + 1) % len(lat)]
                Xaug.append(ae.decoder(z.unsqueeze(0)).squeeze(0).cpu().numpy())
                yaug.append(cls); cnt += 1; i += 2
    if Xaug:
        Xaug = sc.inverse_transform(np.array(Xaug, dtype=np.float32))
        X_out = np.vstack([X_raw, Xaug]); y_out = np.concatenate([y, np.array(yaug)])
    else:
        X_out, y_out = X_raw.copy(), y.copy()
    return X_out.astype(np.float32), y_out.astype(np.int64)


def build_cache(sensor, target, seed):
    s_long = SENSOR_LONG[sensor]
    df = pd.read_csv(DATA / 'train_test_split' / f'train_{s_long}.csv')
    feat = [c for c in df.columns if c.startswith('feat_')]
    X = df[feat].values.astype(np.float32)
    y = LabelEncoder().fit(CLASS_NAMES).transform(df['label'].values)
    fold = df['fold'].values
    out_dir = DATA / 'augmentation' / f'cache_{s_long}'
    out_dir.mkdir(parents=True, exist_ok=True)
    for k in range(N_FOLDS):
        m = fold != k
        Xa, ya = augment_ae(X[m], y[m], target=target, seed=seed)
        np.savez_compressed(out_dir / f'ae_fold{k}.npz', X=Xa, y=ya)
        print(f'  ae fold{k}: in {m.sum()} -> {Xa.shape} {np.bincount(ya)}')
    Xa, ya = augment_ae(X, y, target=target, seed=seed)
    np.savez_compressed(out_dir / 'ae_full.npz', X=Xa, y=ya)
    print(f'  ae full: in {len(X)} -> {Xa.shape} -> {out_dir}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sensor', choices=['sw', 'vn'], required=True)
    ap.add_argument('--target', type=int, default=300, help='target samples per class')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--build_cache', action='store_true',
                    help='build leakage-safe per-fold + full npz caches')
    args = ap.parse_args()
    if args.build_cache:
        build_cache(args.sensor, args.target, args.seed)
    else:
        ap.error('use --build_cache (v2 uses fold-wise npz caches, not a single CSV)')
