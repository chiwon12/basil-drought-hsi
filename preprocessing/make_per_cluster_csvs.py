# -*- coding: utf-8 -*-
"""
Generate per-mode train/test CSVs for Part A.

Modes:
  all_leaf : mean spectrum over all leaf pixels (cluster_label != 255)
  c0 / c1 / c2 : mean spectrum over only that cluster's pixels

Sources (set with the env vars below — large binaries are not bundled):
  - sample list  : data/train_test_split/{train,test}_{swir,vnir}.csv
  - cube         : $CUBE_DIR_SW, $CUBE_DIR_VN (cut-reflectance cubes)
  - cluster label: $LABEL_DIR_SW, $LABEL_DIR_VN (refined K=3 label NPYs)
  - wavelengths  : $WL_SW, $WL_VN  (1D numpy arrays of band centers)

For the representative subset, point CUBE_DIR_* at zenodo_data/cube_samples/.

Output:
  data/per_cluster_split/<mode>_{train,test}_{swir,vnir}.csv

Rows with missing cube/label/cluster (e.g. cluster k pixels absent) get NaN spectra
and are dropped with a warning.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / 'data'
SPLIT = DATA / 'train_test_split'
OUT = DATA / 'per_cluster_split'
OUT.mkdir(parents=True, exist_ok=True)

SENSOR_LONG = {'sw': 'swir', 'vn': 'vnir'}


def _env_path(name: str) -> Path:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f'environment variable {name} is required')
    return Path(v)


CUBE_DIR = {'sw': _env_path('CUBE_DIR_SW'), 'vn': _env_path('CUBE_DIR_VN')}
LABEL_DIR = {'sw': _env_path('LABEL_DIR_SW'), 'vn': _env_path('LABEL_DIR_VN')}
WL_FILE = {'sw': _env_path('WL_SW'), 'vn': _env_path('WL_VN')}


def extract_mode_means(cluster_sample_id: str, sensor: str):
    """Return dict: mode -> mean spectrum (np.ndarray) or None if missing."""
    cube_p = CUBE_DIR[sensor] / f'{cluster_sample_id}_cube.npy'
    label_p = LABEL_DIR[sensor] / f'{cluster_sample_id}_refined.npy'
    if not cube_p.exists() or not label_p.exists():
        return None
    cube = np.load(cube_p, mmap_mode='r')
    label = np.load(label_p)
    if label.shape != cube.shape[:2]:
        return None
    flat_cube = cube.reshape(-1, cube.shape[2])
    flat_lab = label.ravel()
    out = {}
    leaf_mask = flat_lab != 255
    out['all_leaf'] = flat_cube[leaf_mask].mean(axis=0) if leaf_mask.any() else None
    for k in (0, 1, 2):
        sel = flat_lab == k
        out[f'c{k}'] = flat_cube[sel].mean(axis=0) if sel.any() else None
    return out


def main():
    for sensor in ['sw', 'vn']:
        s_long = SENSOR_LONG[sensor]
        wl = np.load(WL_FILE[sensor])
        feat_cols = [f'feat_{w:.2f}' for w in wl]

        for split_name in ['train', 'test']:
            src = pd.read_csv(SPLIT / f'{split_name}_{s_long}.csv')
            rows_by_mode = {'all_leaf': [], 'c0': [], 'c1': [], 'c2': []}
            skipped = {'no_label_or_cube': 0, 'c0_missing': 0, 'c1_missing': 0,
                       'c2_missing': 0, 'all_missing': 0}

            for _, r in src.iterrows():
                csid = r['cluster_sample_id']
                meta = {'sample_id': r['sample_id'], 'cluster_sample_id': csid,
                        'label': r['label']}
                if pd.isna(csid):
                    skipped['no_label_or_cube'] += 1
                    continue
                means = extract_mode_means(csid, sensor)
                if means is None:
                    skipped['no_label_or_cube'] += 1
                    continue
                for mode in ('all_leaf', 'c0', 'c1', 'c2'):
                    spec = means[mode]
                    if spec is None:
                        skipped[f'{mode}_missing' if mode != 'all_leaf' else 'all_missing'] += 1
                        continue
                    row = {**meta}
                    for i, c in enumerate(feat_cols):
                        row[c] = float(spec[i])
                    rows_by_mode[mode].append(row)

            for mode, rows in rows_by_mode.items():
                df = pd.DataFrame(rows)
                out_p = OUT / f'{mode}_{split_name}_{s_long}.csv'
                cols_save = ['sample_id', 'cluster_sample_id'] + feat_cols + ['label']
                df[cols_save].to_csv(out_p, index=False)
                lbl = dict(df['label'].value_counts()) if len(df) else {}
                print(f'  [{sensor} {split_name} {mode}]  rows={len(df)}  labels={lbl}')
            print(f'  [{sensor} {split_name}] skipped: {skipped}')
        print()


if __name__ == '__main__':
    main()
