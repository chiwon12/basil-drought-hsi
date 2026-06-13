# -*- coding: utf-8 -*-
"""
Build the manuscript result tables from the experiment outputs.

Reads the aggregate produced by training/run_single_experiment.py
(`results/ALL_RESULTS_combined.csv` + per-combo `results/per_combo/<combo>/
test_metrics.json`) and the static leaky-baseline results
(`data/before_leaky_sw.csv`), and writes, under `results/tables/`:

  master_pivot_{test_acc,weighted_f1}.csv  model × (norm·data)
  Table3_main_raw.csv                       raw data, model × norm (acc + wF1)
  Table4_augmentation_snv.csv               SNV, model × data (acc)
  Table5_best_per_model.csv                 best config per model
  before_after.csv                          v1 (leaky) vs v2 (plant-aware) Δ per combo
  final_hyperparameters.csv                 adopted Optuna best params
  S2..S10_<norm>_<data>.csv                 per-condition all-model metrics (supplementary)
  SUMMARY.md                                headline summary

Run after the sweep (default sensor = sw, the manuscript's SWIR analysis):
  python src/analysis/build_tables.py
  python src/analysis/build_tables.py --sensor sw --results_root results
"""
import argparse
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
NORMS = ['none', 'snv', 'msc']
DATAS = ['raw', 'ae', 'gan']
MODEL_ORDER = ['knn', 'svm', 'rf', 'xgb', 'cnn']
M2LONG = {'knn': 'KNN', 'svm': 'SVM', 'rf': 'RandomForest', 'xgb': 'XGBoost', 'cnn': 'CNN'}
L2SHORT = {v: k for k, v in M2LONG.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sensor', default='sw', choices=['sw', 'vn'])
    ap.add_argument('--results_root', default=None, help='default: repo/results')
    ap.add_argument('--before_csv', default=None,
                    help='static leaky-baseline results (default: data/before_leaky_sw.csv)')
    args = ap.parse_args()

    results = Path(args.results_root) if args.results_root else REPO / 'results'
    T = results / 'tables'; T.mkdir(parents=True, exist_ok=True)
    before_csv = Path(args.before_csv) if args.before_csv else REPO / 'data' / 'before_leaky_sw.csv'

    after = pd.read_csv(results / 'ALL_RESULTS_combined.csv')
    after['model'] = after['model'].str.lower()
    if 'sensor' in after.columns:
        after = after[after['sensor'].str.lower() == args.sensor].copy()
    print(f'after combos: {len(after)}/45')

    def pivot(metric):
        after['cond'] = after['norm'] + '_' + after['data']
        p = after.pivot_table(index='model', columns='cond', values=metric)
        return p.reindex(index=MODEL_ORDER, columns=[f'{n}_{d}' for n in NORMS for d in DATAS])
    pivot('test_acc').to_csv(T / 'master_pivot_test_acc.csv')
    pivot('test_weighted_f1').to_csv(T / 'master_pivot_weighted_f1.csv')

    # Table 3: raw, model × norm
    raw = after[after.data == 'raw']
    t3 = raw.pivot_table(index='model', columns='norm', values='test_acc').reindex(MODEL_ORDER, columns=NORMS)
    t3f = raw.pivot_table(index='model', columns='norm', values='test_weighted_f1').reindex(MODEL_ORDER, columns=NORMS)
    t3.columns = [f'acc_{c}' for c in t3.columns]
    t3f.columns = [f'wF1_{c}' for c in t3f.columns]
    pd.concat([t3, t3f], axis=1).to_csv(T / 'Table3_main_raw.csv')

    # Table 4: SNV, model × data
    snv = after[after.norm == 'snv']
    snv.pivot_table(index='model', columns='data', values='test_acc').reindex(
        MODEL_ORDER, columns=DATAS).to_csv(T / 'Table4_augmentation_snv.csv')

    # Table 5: best per model
    rows = []
    for m in MODEL_ORDER:
        sub = after[after.model == m]
        if not len(sub):
            continue
        b = sub.loc[sub.test_weighted_f1.idxmax()]
        rows.append(dict(model=M2LONG[m], best_norm=b.norm, best_data=b.data,
                         test_acc=round(b.test_acc, 4), test_weighted_f1=round(b.test_weighted_f1, 4),
                         test_macro_f1=round(b.test_macro_f1, 4), test_auc=round(b.test_auc, 4),
                         cv_weighted_f1=round(b.cv_weighted_f1_mean, 4)))
    pd.DataFrame(rows).to_csv(T / 'Table5_best_per_model.csv', index=False)

    # before/after
    merged = None
    if before_csv.exists():
        before = pd.read_csv(before_csv)
        before['model'] = before['model'].map(lambda x: L2SHORT.get(x, x.lower()))
        bm = before.rename(columns={'test_acc': 'before_test_acc', 'test_f1': 'before_test_f1',
                                    'cv_acc': 'before_cv_acc'})
        am = after.rename(columns={'test_acc': 'after_test_acc', 'test_weighted_f1': 'after_test_wf1'})
        merged = bm[['model', 'norm', 'data', 'before_cv_acc', 'before_test_acc', 'before_test_f1']].merge(
            am[['model', 'norm', 'data', 'cv_weighted_f1_mean', 'after_test_acc', 'after_test_wf1']],
            on=['model', 'norm', 'data'], how='outer')
        merged['delta_test_acc'] = (merged.after_test_acc - merged.before_test_acc).round(4)
        merged['model'] = merged['model'].map(M2LONG)
        merged = merged.sort_values(['model', 'norm', 'data'])
        merged.to_csv(T / 'before_after.csv', index=False)
    else:
        print(f'[warn] {before_csv} not found - skipping before/after table')

    # final hyperparameters
    hp = []
    for _, r in after.iterrows():
        mp = results / 'per_combo' / f"{args.sensor}_{r.norm}_{r.data}_{r.model}" / 'test_metrics.json'
        params = json.loads(mp.read_text())['best_params'] if mp.exists() else {}
        hp.append(dict(model=M2LONG[r.model], norm=r.norm, data=r.data,
                       test_acc=round(r.test_acc, 4), best_params=json.dumps(params)))
    pd.DataFrame(hp).sort_values(['model', 'norm', 'data']).to_csv(
        T / 'final_hyperparameters.csv', index=False)

    # per-condition supplementary tables (S2..S10)
    si = 2
    for n in NORMS:
        for d in DATAS:
            sub = after[(after.norm == n) & (after.data == d)].copy()
            if not len(sub):
                continue
            sub['model'] = sub['model'].map(M2LONG)
            sub = sub.set_index('model').reindex([M2LONG[m] for m in MODEL_ORDER])
            cols = ['cv_weighted_f1_mean', 'cv_weighted_f1_std', 'test_acc',
                    'test_weighted_f1', 'test_macro_f1', 'test_auc']
            sub[cols].round(4).to_csv(T / f'S{si}_{n}_{d}.csv')
            si += 1

    # SUMMARY.md
    lines = ['# Plant-Aware re-run result summary\n',
             f'- combos: {len(after)}/45 (3 norm x 3 data x 5 model), sensor={args.sensor}',
             '- split: plant-level 80:20 (class-balanced, 0 plant leakage) / StratifiedGroupKFold(5)',
             '- selection: mean weighted-F1; HP = Optuna TPE 200 trials (ML & CNN), no pruner',
             '- leakage control: scaler/aug/MSC-ref fit on fold-train only; val & test = raw\n',
             '## Table 3 - raw data, test accuracy (model x norm)', t3.round(4).to_markdown(),
             '\n## Table 5 - best config per model', pd.DataFrame(rows).to_markdown(index=False)]
    if merged is not None:
        ov = merged.dropna(subset=['before_test_acc', 'after_test_acc'])
        if len(ov):
            lines += ['\n## Before(leaky) -> After(plant-aware)',
                      f'- mean test_acc: before {ov.before_test_acc.mean():.3f} -> '
                      f'after {ov.after_test_acc.mean():.3f} (mean Δ {ov.delta_test_acc.mean():+.3f})',
                      f'- combos that dropped: {(ov.delta_test_acc < 0).sum()}/{len(ov)}']
    (T / 'SUMMARY.md').write_text('\n'.join(str(x) for x in lines))
    print('tables ->', T)


if __name__ == '__main__':
    main()
