# -*- coding: utf-8 -*-
"""
Rebuild SW/VN train/test split using the notebook procedure applied to
pivot_labeled_<sensor>_data.csv, while preserving sample_id (and adding
the cluster_sample_id derived from plate-quad → plant_id mapping).

Procedure:
  - Input: pivot_labeled_<sensor>_data.csv (has sample_id + feat_* + label)
  - 80/20 train_test_split, stratified by label, random_state=42
  - Output: data/train_test_split/{train,test}_{swir,vnir}.csv
            (columns: sample_id, cluster_sample_id, feat_*, label)

Plate-quad -> plant_id mapping:
  top_left=1st, top_right=2nd, bottom_left=3rd, bottom_right=4th of plate range.

Note: the source pivot CSVs are not bundled in this release; pass --pivot_dir
to point this script at your local copy. The output split CSVs we used are
already provided in data/train_test_split/.
"""
import argparse
import re
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / 'data' / 'train_test_split'
OUT_DIR.mkdir(parents=True, exist_ok=True)
SENSOR_LONG = {'sw': 'swir', 'vn': 'vnir'}

QUAD_OFFSET = {'top_left': 0, 'top_right': 1, 'bottom_left': 2, 'bottom_right': 3}


def pivot_to_cluster_id(pivot_sid: str, label: str, sensor: str):
    sid = re.sub(r'_+', '_', pivot_sid)
    m = re.search(r'_cube_(top_left|top_right|bottom_left|bottom_right)_', sid)
    if not m:
        return None
    quad = m.group(1)
    head = sid[:m.start()]
    parts = head.split('_')
    if len(parts) < 7:
        return None
    plate_a, plate_b, date, kind, band = parts[2], parts[3], parts[4], parts[5], parts[6]
    a_pre, a_n = plate_a.split('-'); b_pre, b_n = plate_b.split('-')
    if a_pre != b_pre:
        return None
    start, end = int(a_n), int(b_n)
    if end - start + 1 != 4:
        return None
    plant_n = start + QUAD_OFFSET[quad]
    plant_id = f'{a_pre}-{plant_n}'
    if sensor == 'sw':
        return f'SW_{label}_{plant_id}_{date}_{kind}_{band}'
    return f'VN_{label}_BS_{plant_id}_{date}_{kind}_{band}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pivot_dir', required=True,
                    help='directory containing pivot_labeled_{sw,vn}_data.csv')
    args = ap.parse_args()

    pivot_dir = Path(args.pivot_dir)
    ts = time.strftime('%Y%m%d_%H%M%S')
    for sensor in ['sw', 'vn']:
        s_long = SENSOR_LONG[sensor]
        pivot_path = pivot_dir / f'pivot_labeled_{sensor}_data.csv'
        df = pd.read_csv(pivot_path)
        feat_cols = [c for c in df.columns if c.startswith('feat_')]
        df['cluster_sample_id'] = [pivot_to_cluster_id(s, l, sensor)
                                    for s, l in zip(df['sample_id'], df['label'])]
        n_built = df['cluster_sample_id'].notna().sum()

        train, test = train_test_split(
            df, test_size=0.2, stratify=df['label'], random_state=42)

        for split_name, split_df in [('train', train), ('test', test)]:
            out_p = OUT_DIR / f'{split_name}_{s_long}.csv'
            if out_p.exists():
                bak = out_p.with_suffix(f'.bak_{ts}.csv')
                out_p.rename(bak)
            cols_save = ['sample_id', 'cluster_sample_id'] + feat_cols + ['label']
            split_df[cols_save].to_csv(out_p, index=False)
            print(f'  [{sensor}] {split_name}={len(split_df)}  -> {out_p}')

        print(f'[{sensor}] pivot rows={len(df)}, cluster_sample_id built={n_built}, '
              f'train labels={dict(train["label"].value_counts())}, '
              f'test labels={dict(test["label"].value_counts())}')


if __name__ == '__main__':
    main()
