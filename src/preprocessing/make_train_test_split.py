# -*- coding: utf-8 -*-
"""
Plant-aware train/test split + StratifiedGroupKFold(5).

Leakage-safe split design
-------------------------
Splitting on individual leaf spectra leaks information: the same physical plant
is imaged on several dates, so rows from one plant could land on both sides of
the split (and across CV folds), inflating the cross-validated scores. This
split operates at the **plant** level:

  * external hold-out : 80:20 over *plants*, class-stratified (per-class random
    draw so every class keeps the 20% test share);
  * internal CV       : StratifiedGroupKFold(5) over the training plants, so the
    same plant never appears in both a fold's train and validation parts.

Plant key (derived from `cluster_sample_id`)
  * SWIR : ``SW_<label>_<tray-plant>_<date>_...``  -> plant = ``<label>_<tray-plant>``
  * VNIR : ``VN_<label>_BS_<tray-plant>_<date>_...`` -> plant = ``<label>_<tray-plant>``

Input : pivot_labeled_<sensor>_data.csv  (sample_id + feat_* + label)
Output: data/train_test_split/{train,test,full_pool,manifest}_{swir,vnir}.csv
        (meta columns: sample_id, cluster_sample_id, plant, date, label, split, fold;
         fold = -1 for test rows)  plus  split_report_<sensor>.json

The source pivot CSVs are not bundled; pass --pivot_dir to point at your local
copy. The split CSVs we used are already provided under data/train_test_split/.
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / 'data' / 'train_test_split'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SENSOR_LONG = {'sw': 'swir', 'vn': 'vnir'}
CLASS_NAMES = ['DR', 'MD', 'SD', 'WW']
QUAD_OFFSET = {'top_left': 0, 'top_right': 1, 'bottom_left': 2, 'bottom_right': 3}
SEED, N_FOLDS, TEST_FRAC = 42, 5, 0.2


def parse_pivot_sid(pivot_sid: str):
    """Parse a pivot sample_id into a dict with plate pair, date, quadrant,
    plate prefix and the plate start/end plant numbers. Returns None if it does
    not match the plate pattern (used to trigger recovery for VNIR batches)."""
    sid = re.sub(r'_+', '_', pivot_sid)
    m = re.search(r'_cube_(top_left|top_right|bottom_left|bottom_right)_', sid)
    if not m:
        return None
    quad = m.group(1)
    parts = sid[:m.start()].split('_')
    if len(parts) < 7:
        return None
    plate_a, plate_b, date, kind, band = parts[2], parts[3], parts[4], parts[5], parts[6]
    a_pre, a_n = plate_a.split('-')
    b_pre, b_n = plate_b.split('-')
    if a_pre != b_pre or int(b_n) - int(a_n) + 1 != 4:
        return None
    return dict(pair=f'{plate_a}_{plate_b}', date=date, quad=quad, kind=kind, band=band,
                prefix=a_pre, start=int(a_n), end=int(b_n))


def build_meta(df: pd.DataFrame, sensor: str):
    """Add cluster_sample_id / plant / date columns. Recovers rows whose
    sample_id does not parse (VNIR early-date batches) via a learned
    (pair, quadrant) -> (label, plant_id) mapping, mirroring the analysis pipeline."""
    info = [parse_pivot_sid(s) for s in df['sample_id']]

    def cid_of(rec, label):
        if rec is None:
            return None
        plant_id = f"{rec['prefix']}-{rec['start'] + QUAD_OFFSET[rec['quad']]}"
        if sensor == 'sw':
            return f"SW_{label}_{plant_id}_{rec['date']}_{rec['kind']}_{rec['band']}"
        return f"VN_{label}_BS_{plant_id}_{rec['date']}_{rec['kind']}_{rec['band']}"

    df = df.copy()
    df['cluster_sample_id'] = [cid_of(r, l) for r, l in zip(info, df['label'])]

    # learn (pair, quad) -> {(label, plant_id)} from rows that parsed, to recover the rest
    pq2plant = {}
    for rec, label in zip(info, df['label']):
        if rec is not None:
            plant_id = f"{rec['prefix']}-{rec['start'] + QUAD_OFFSET[rec['quad']]}"
            pq2plant.setdefault((rec['pair'], rec['quad']), set()).add((label, plant_id))

    def plant_date(rec, label, cid):
        if cid is not None:
            p = cid.split('_')
            return (f'{p[1]}_{p[2]}', p[3]) if sensor == 'sw' else (f'{p[1]}_{p[3]}', p[4])
        if rec is not None:                                   # recover unparsed cid
            for lab, pid in pq2plant.get((rec['pair'], rec['quad']), set()):
                if lab == label:
                    return f'{label}_{pid}', rec['date']
        return None, None

    pairs = [plant_date(r, l, c) for r, l, c in
             zip(info, df['label'], df['cluster_sample_id'])]
    df['plant'] = [p for p, _ in pairs]
    df['date'] = [d for _, d in pairs]
    return df


def split_and_fold(df: pd.DataFrame):
    """Plant-level 80:20 (class-stratified) + StratifiedGroupKFold(5) on train."""
    plants = sorted(df['plant'].unique())
    plant_cls = df.groupby('plant')['label'].first().to_dict()
    assert (df.groupby('plant')['label'].nunique() == 1).all(), 'plant spans >1 label'

    rng = np.random.RandomState(SEED)
    test_plants = set()
    for cls in CLASS_NAMES:
        cps = sorted([p for p in plants if plant_cls[p] == cls])
        n_test = round(len(cps) * TEST_FRAC)
        test_plants.update(rng.choice(cps, size=n_test, replace=False).tolist())
    df = df.copy()
    df['split'] = np.where(df['plant'].isin(test_plants), 'test', 'train')

    tr = df[df['split'] == 'train'].reset_index()
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    df['fold'] = -1
    for fold, (_, vi) in enumerate(sgkf.split(tr, tr['label'], tr['plant'])):
        df.loc[tr.loc[vi, 'index'].values, 'fold'] = fold
    return df


def verify(df: pd.DataFrame, feat_cols):
    rep = {'seed': SEED, 'n_folds': N_FOLDS, 'n_features': len(feat_cols),
           'total_rows': int(len(df)), 'total_plants': int(df['plant'].nunique()),
           'recovered_nan_cid': int(df['cluster_sample_id'].isna().sum())}
    per = {}
    for cls in CLASS_NAMES:
        per[cls] = dict(
            train_plant=int(df[(df.label == cls) & (df.split == 'train')]['plant'].nunique()),
            test_plant=int(df[(df.label == cls) & (df.split == 'test')]['plant'].nunique()),
            train_img=int(((df.label == cls) & (df.split == 'train')).sum()),
            test_img=int(((df.label == cls) & (df.split == 'test')).sum()))
    rep['per_class'] = per
    rep['train_img'] = int((df.split == 'train').sum())
    rep['test_img'] = int((df.split == 'test').sum())
    leak = set(df[df.split == 'train'].plant) & set(df[df.split == 'test'].plant)
    rep['plant_leakage'] = len(leak)
    fps = [set(df[(df.split == 'train') & (df.fold == k)].plant) for k in range(N_FOLDS)]
    rep['fold_plant_leakage'] = bool(any(fps[i] & fps[j]
                                         for i in range(N_FOLDS) for j in range(i + 1, N_FOLDS)))
    rep['folds'] = [dict(fold=k, plant=int(s.plant.nunique()), img=int(len(s)),
                         dist={c: int((s.label == c).sum()) for c in CLASS_NAMES})
                    for k, s in ((k, df[(df.split == 'train') & (df.fold == k)])
                                 for k in range(N_FOLDS))]
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pivot_dir', required=True,
                    help='directory containing pivot_labeled_{sw,vn}_data.csv')
    ap.add_argument('--sensors', nargs='+', default=['sw', 'vn'], choices=['sw', 'vn'])
    args = ap.parse_args()
    pivot_dir = Path(args.pivot_dir)
    ts = time.strftime('%Y%m%d_%H%M%S')

    for sensor in args.sensors:
        s_long = SENSOR_LONG[sensor]
        df = pd.read_csv(pivot_dir / f'pivot_labeled_{sensor}_data.csv')
        feat_cols = [c for c in df.columns if c.startswith('feat_')]
        df = build_meta(df, sensor)
        df = split_and_fold(df)

        meta = ['sample_id', 'cluster_sample_id', 'plant', 'date', 'label', 'split', 'fold']
        out = {
            f'full_pool_{s_long}.csv': df[meta + feat_cols],
            f'train_{s_long}.csv': df[df.split == 'train'][meta + feat_cols],
            f'test_{s_long}.csv': df[df.split == 'test'][meta + feat_cols],
            f'manifest_{s_long}.csv': df[meta],
        }
        for name, sub in out.items():
            p = OUT_DIR / name
            if p.exists():
                p.rename(p.with_suffix(f'.bak_{ts}.csv'))
            sub.to_csv(p, index=False)

        rep = verify(df, feat_cols)
        json.dump(rep, open(OUT_DIR / f'split_report_{sensor}.json', 'w'),
                  indent=2, ensure_ascii=False)
        print(f"[{sensor}] rows={rep['total_rows']} plants={rep['total_plants']} "
              f"train/test img={rep['train_img']}/{rep['test_img']} "
              f"plant_leak={rep['plant_leakage']} fold_leak={rep['fold_plant_leakage']}")


if __name__ == '__main__':
    main()
