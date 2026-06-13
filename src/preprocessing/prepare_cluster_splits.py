# -*- coding: utf-8 -*-
"""
Re-assign per-cluster spectra to the plant-aware split/fold.

`make_per_cluster_csvs.py` produces per-cluster mean-spectrum tables
(``data/per_cluster_split/c{k}_{train,test}_{swir,vnir}.csv``). To keep the
leakage fix consistent across all scopes, the cluster tables must use the SAME
plant-level split and StratifiedGroupKFold folds as the all-leaf data instead of
their own train/test boundary.

This script drops the old train/test labels, pools each cluster's rows, joins
them to the plant-aware manifest (``data/train_test_split/manifest_<sensor>.csv``)
on ``sample_id``, and writes split CSVs with the identical columns/structure as
the all-leaf split:

  data/per_cluster_split/c{k}/{train,test,full_pool,manifest}_{swir,vnir}.csv

The cluster-mode experiments then reuse the standard runner by pointing at the
cluster splits directory:

  python augmentation/augment_with_autoencoder.py --sensor sw --build_cache \
      ... (or run raw-only)
  python training/run_single_experiment.py --sensor sw --norm none --data raw \
      --model all --splits_dir data/per_cluster_split/c0 \
      --out_root results/cluster/c0

Usage:
  python prepare_cluster_splits.py [--sensors sw vn] [--clusters 0 1 2]
"""
import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / 'data'
SPLIT = DATA / 'train_test_split'
PCS = DATA / 'per_cluster_split'
SENSOR_LONG = {'sw': 'swir', 'vn': 'vnir'}
META = ['sample_id', 'cluster_sample_id', 'plant', 'date', 'label', 'split', 'fold']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sensors', nargs='+', default=['sw', 'vn'], choices=['sw', 'vn'])
    ap.add_argument('--clusters', nargs='+', type=int, default=[0, 1, 2])
    args = ap.parse_args()

    for sensor in args.sensors:
        prefix = SENSOR_LONG[sensor]
        man = pd.read_csv(SPLIT / f'manifest_{prefix}.csv')   # sample_id -> plant/split/fold
        for k in args.clusters:
            parts = [pd.read_csv(PCS / f'c{k}_{sp}_{prefix}.csv')
                     for sp in ('train', 'test')
                     if (PCS / f'c{k}_{sp}_{prefix}.csv').exists()]
            if not parts:
                print(f'[skip] {prefix} c{k}: source per-cluster CSVs not found')
                continue
            pool = pd.concat(parts, ignore_index=True)
            feat = [c for c in pool.columns if c.startswith('feat_')]
            pool = pool.dropna(subset=feat)                   # drop samples with no cluster-k pixels
            merged = pool[['sample_id'] + feat].merge(man, on='sample_id', how='inner')

            out = PCS / f'c{k}'; out.mkdir(parents=True, exist_ok=True)
            cols = META + feat
            merged[cols].to_csv(out / f'full_pool_{prefix}.csv', index=False)
            merged[merged.split == 'train'][cols].to_csv(out / f'train_{prefix}.csv', index=False)
            merged[merged.split == 'test'][cols].to_csv(out / f'test_{prefix}.csv', index=False)
            merged[META].to_csv(out / f'manifest_{prefix}.csv', index=False)

            leak = set(merged[merged.split == 'train'].plant) & set(merged[merged.split == 'test'].plant)
            print(f"{prefix} c{k}: train={int((merged.split=='train').sum())} "
                  f"test={int((merged.split=='test').sum())} "
                  f"folds={merged[merged.split=='train'].fold.nunique()} "
                  f"plant_leak={len(leak)} {'OK' if not leak else 'LEAK!'}")


if __name__ == '__main__':
    main()
