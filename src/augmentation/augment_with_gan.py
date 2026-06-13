# -*- coding: utf-8 -*-
"""GAN-based per-class spectral augmentation (PyTorch).

Two changes vs the first release:
  1. PyTorch port of the original Keras per-class GAN (identical architecture and
     hyper-parameters: G Linear(64)->Linear(D) from latent=100, D Linear(64)->
     Linear(1)+Sigmoid, BCE, 3000 epochs, batch 32, Adam; one GAN per class,
     sampling up to `target` per class).
  2. **Leakage-safe, fold-wise caches.** `--build_cache` reads the split CSV
     (with its `fold` column) and writes, per CV fold k, augmentation built from
     rows with ``fold != k`` only, plus a `full` cache from the whole new-train.
     Validation/test rows are never augmented.

`augment_gan(X_raw, y, target, seed)` is the importable library function used by
both this CLI and the CV loop in training/run_single_experiment.py.

Usage:
  python augment_with_gan.py --sensor sw --build_cache
  python augment_with_gan.py --sensor vn --build_cache
Output: data/augmentation/cache_{swir,vnir}/gan_fold{0..4}.npz, gan_full.npz
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


class _G(nn.Module):
    def __init__(self, d, latent=100):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent, 64), nn.ReLU(), nn.Linear(64, d))

    def forward(self, z):
        return self.net(z)


class _D(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x):
        return self.net(x)


def augment_gan(X_raw, y, target=300, seed=42, epochs=3000, latent=100, bs=32):
    """X_raw (n, D) reflectance, y (n,) int. Returns (X_out, y_out) including
    originals, each class padded to `target` by a per-class GAN. All fitting
    happens on the rows passed in — never on val/test."""
    torch.manual_seed(seed); np.random.seed(seed)
    sc = StandardScaler().fit(X_raw)
    Xs = sc.transform(X_raw).astype(np.float32)
    d = Xs.shape[1]
    bce = nn.BCELoss()
    Xaug, yaug = [], []
    for cls in np.unique(y):
        cx = Xs[y == cls]
        n_gen = target - len(cx)
        if n_gen <= 0:
            continue
        G, D = _G(d, latent).to(DEVICE), _D(d).to(DEVICE)
        oG, oD = torch.optim.Adam(G.parameters()), torch.optim.Adam(D.parameters())
        cxT = torch.from_numpy(cx).to(DEVICE)
        for _ in range(epochs):
            real = cxT[np.random.randint(0, len(cx), bs)]
            with torch.no_grad():
                fake = G(torch.randn(bs, latent, device=DEVICE))
            oD.zero_grad()
            (bce(D(real), torch.ones(bs, 1, device=DEVICE)) +
             bce(D(fake), torch.zeros(bs, 1, device=DEVICE))).backward()
            oD.step()
            oG.zero_grad()
            bce(D(G(torch.randn(bs, latent, device=DEVICE))),
                torch.ones(bs, 1, device=DEVICE)).backward()
            oG.step()
        G.eval()
        with torch.no_grad():
            gen = G(torch.randn(n_gen, latent, device=DEVICE)).cpu().numpy()
        Xaug.append(sc.inverse_transform(gen)); yaug += [cls] * n_gen
    if Xaug:
        X_out = np.vstack([X_raw] + Xaug); y_out = np.concatenate([y, np.array(yaug)])
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
        Xa, ya = augment_gan(X[m], y[m], target=target, seed=seed)
        np.savez_compressed(out_dir / f'gan_fold{k}.npz', X=Xa, y=ya)
        print(f'  gan fold{k}: in {m.sum()} -> {Xa.shape} {np.bincount(ya)}')
    Xa, ya = augment_gan(X, y, target=target, seed=seed)
    np.savez_compressed(out_dir / 'gan_full.npz', X=Xa, y=ya)
    print(f'  gan full: in {len(X)} -> {Xa.shape} -> {out_dir}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sensor', choices=['sw', 'vn'], required=True)
    ap.add_argument('--target', type=int, default=300)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--build_cache', action='store_true',
                    help='build leakage-safe per-fold + full npz caches')
    args = ap.parse_args()
    if args.build_cache:
        build_cache(args.sensor, args.target, args.seed)
    else:
        ap.error('use --build_cache (v2 uses fold-wise npz caches, not a single CSV)')
