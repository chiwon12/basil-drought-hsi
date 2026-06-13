# -*- coding: utf-8 -*-
"""
One (sensor, norm, data, model) experiment per invocation.

Leakage-safe pipeline:
  1. Load the plant-aware split CSV (carries the StratifiedGroupKFold `fold`
     column). Validation/test rows are always raw.
  2. For each CV fold k: training rows are raw (data=raw) or the fold's own
     augmentation cache `ae/gan_fold{k}.npz` (built only from fold!=k rows);
     normalization + StandardScaler are fit on the fold's train part only.
  3. Optuna TPE search (200 trials, no pruner), selection = mean weighted-F1
     across folds.
  4. Re-run 5-fold CV with the best params -> per-fold metrics + (CNN) curves.
  5. Final fit on the whole new-train (raw, or `*_full.npz` augmentation) and
     evaluate on the held-out test set, which is ALWAYS raw and never used to
     pick epochs/params. For the CNN the final epoch count is the mean best
     epoch measured during CV.

Inputs  (relative to repo root):
  data/train_test_split/{train,test}_{swir,vnir}.csv
  data/augmentation/cache_{swir,vnir}/{ae,gan}_{fold0..4,full}.npz   (data!=raw)
Outputs:
  results/per_combo/<combo>/{test_metrics.json, train_cv_per_fold.csv,
    optuna_trials.csv, confusion_matrix.png, roc_curve.png,
    per_class_metrics.csv, training_curves.png (CNN)}
  results/optuna_db/<combo>.db ; results/ALL_RESULTS_combined.csv

Usage:
  python run_single_experiment.py --sensor sw --norm none --data raw --model knn
  python run_single_experiment.py --sensor vn --norm snv --data ae --model all --gpu 0
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch  # noqa: F401  — must import before xgboost (libnccl symbol clash on some installs)
import matplotlib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             confusion_matrix, f1_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import LabelEncoder, label_binarize

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

REPO = Path(__file__).resolve().parents[2]          # repo root
SRC = Path(__file__).resolve().parents[1]            # src/
for sub in ('normalization', 'models'):
    sys.path.insert(0, str(SRC / sub))
from spectral_normalization import normalize_train_test          # noqa: E402
from cnn_1d_model import cnn_fit_fixed, cnn_train_eval, suggest_cnn  # noqa: E402 (torch before xgb)
from classical_ml_models import build_ml, suggest_ml             # noqa: E402

DATA = REPO / 'data'
CLASS_NAMES = ['DR', 'MD', 'SD', 'WW']
N_CLASSES = 4
N_FOLDS = 5
SENSOR_LONG = {'sw': 'swir', 'vn': 'vnir'}


# ── fold data (raw or cached augmentation) ──
def load_folds(data, X, y, fold, cache_dir):
    folds = []
    for k in range(N_FOLDS):
        va = fold == k
        Xva, yva = X[va], y[va]                       # val always raw
        if data == 'raw':
            Xtr, ytr = X[~va], y[~va]
        else:
            z = np.load(cache_dir / f'{data}_fold{k}.npz')
            Xtr, ytr = z['X'], z['y']                 # augmented from fold!=k only
        folds.append((Xtr.astype(np.float32), ytr, Xva.astype(np.float32), yva))
    return folds


def load_full_train(data, X, y, cache_dir):
    if data == 'raw':
        return X.astype(np.float32), y
    z = np.load(cache_dir / f'{data}_full.npz')
    return z['X'].astype(np.float32), z['y']


def cv_weighted_f1(model, params, folds, norm, seed, cnn_ep, cnn_pat, collect=False):
    rows, hists = [], []
    for k, (Xtr, ytr, Xva, yva) in enumerate(folds):
        Xt, Xv = normalize_train_test(Xtr, Xva, norm)
        if model == 'cnn':
            pred, prob, h = cnn_train_eval(Xt, ytr, Xv, yva, params, N_CLASSES,
                                           cnn_ep, cnn_pat, seed, return_history=True)
            hists.append(h)
        else:
            mm = build_ml(model, params, N_CLASSES, seed); mm.fit(Xt, ytr)
            pred = mm.predict(Xv)
            prob = None if model == 'svm' else mm.predict_proba(Xv)  # svm proba off for speed
        wf1 = f1_score(yva, pred, average='weighted')
        if collect:
            try:
                au = roc_auc_score(label_binarize(yva, classes=range(N_CLASSES)),
                                   prob, multi_class='ovr')
            except Exception:
                au = float('nan')
            rows.append(dict(fold=k, acc=accuracy_score(yva, pred), weighted_f1=wf1,
                             macro_f1=f1_score(yva, pred, average='macro'), auc=au))
        else:
            rows.append(wf1)
    return (rows, hists) if collect else float(np.mean(rows))


def plot_confusion(cm, labels, title, path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels,
                yticklabels=labels, vmin=0, vmax=1)
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


def plot_roc(y, prob, labels, title, path):
    yb = label_binarize(y, classes=range(len(labels)))
    plt.figure(figsize=(6, 5))
    for i in range(len(labels)):
        fpr, tpr, _ = roc_curve(yb[:, i], prob[:, i])
        plt.plot(fpr, tpr, label=f'{labels[i]} (AUC={auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], 'k--'); plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title(title); plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


def plot_curves(hists, path):
    _, ax = plt.subplots(1, 3, figsize=(15, 4))
    for f, h in enumerate(hists):
        ep = range(1, len(h['tr_loss']) + 1)
        ax[0].plot(ep, h['tr_loss'], label=f'fold{f}')
        ax[1].plot(ep, h['va_loss'], label=f'fold{f}')
        ax[2].plot(ep, h['va_acc'], label=f'fold{f}')
    for a, ti in zip(ax, ['train loss', 'val loss', 'val acc']):
        a.set_title(ti); a.set_xlabel('epoch'); a.legend()
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def run(sensor, norm, data, model, ml_trials, cnn_trials, cnn_ep, cnn_pat, seed, out_root,
        splits_dir=None, cache_dir=None):
    prefix = SENSOR_LONG[sensor]
    splits_dir = Path(splits_dir) if splits_dir else DATA / 'train_test_split'
    cache_dir = Path(cache_dir) if cache_dir else DATA / 'augmentation' / f'cache_{prefix}'
    combo = f'{sensor}_{norm}_{data}_{model}'
    od = out_root / 'per_combo' / combo; od.mkdir(parents=True, exist_ok=True)
    mpath = od / 'test_metrics.json'
    if mpath.exists():
        print(f'[skip] {combo}'); return
    t0 = time.time()
    le = LabelEncoder().fit(CLASS_NAMES)
    tr = pd.read_csv(splits_dir / f'train_{prefix}.csv')
    te = pd.read_csv(splits_dir / f'test_{prefix}.csv')
    feat = [c for c in tr.columns if c.startswith('feat_')]
    Xtr_all = tr[feat].values.astype(np.float32); ytr_all = le.transform(tr['label'].values)
    fold = tr['fold'].values
    Xte = te[feat].values.astype(np.float32); yte = le.transform(te['label'].values)
    folds = load_folds(data, Xtr_all, ytr_all, fold, cache_dir)
    trials = cnn_trials if model == 'cnn' else ml_trials
    print(f'=== {combo} trials={trials} train={len(Xtr_all)} test={len(Xte)} feat={len(feat)} ===',
          flush=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(t):
        params = suggest_cnn(t) if model == 'cnn' else suggest_ml(model, t)
        return cv_weighted_f1(model, params, folds, norm, seed, cnn_ep, cnn_pat)

    # No pruner (NopPruner) per paper Table S1. Per-combo SQLite DB so concurrent
    # runs never race on table creation; resumable.
    dbdir = out_root / 'optuna_db'; dbdir.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(study_name=combo,
                                storage=f"sqlite:///{dbdir / f'{combo}.db'}",
                                load_if_exists=True, direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.NopPruner())
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    best = dict(study.best_params); best_cv = float(study.best_value)
    print(f'  best CV weighted_f1={best_cv:.4f} params={best}', flush=True)
    study.trials_dataframe().to_csv(od / 'optuna_trials.csv', index=False)

    cv_rows, hists = cv_weighted_f1(model, best, folds, norm, seed, cnn_ep, cnn_pat, collect=True)
    pd.DataFrame(cv_rows).to_csv(od / 'train_cv_per_fold.csv', index=False)
    if hists:
        plot_curves(hists, od / 'training_curves.png')

    # Final: fit whole new-train -> evaluate raw hold-out test.
    Xfull, yfull = load_full_train(data, Xtr_all, ytr_all, cache_dir)
    Xf, Xt = normalize_train_test(Xfull, Xte, norm)
    if model == 'cnn':
        best_epochs = (int(round(np.mean([int(np.argmin(h['va_loss'])) + 1 for h in hists])))
                       if hists else cnn_ep)
        pred, prob = cnn_fit_fixed(Xf, yfull, best, N_CLASSES, best_epochs, seed, Xpred=Xt)
    else:
        mm = build_ml(model, best, N_CLASSES, seed, proba=True); mm.fit(Xf, yfull)
        pred = mm.predict(Xt); prob = mm.predict_proba(Xt)
    acc = accuracy_score(yte, pred)
    wf1 = f1_score(yte, pred, average='weighted'); mf1 = f1_score(yte, pred, average='macro')
    try:
        aucv = roc_auc_score(label_binarize(yte, classes=range(N_CLASSES)), prob, multi_class='ovr')
    except Exception:
        aucv = float('nan')
    print(f'  TEST acc={acc:.4f} wF1={wf1:.4f} mF1={mf1:.4f} auc={aucv:.4f}', flush=True)

    cm_norm = confusion_matrix(yte, pred, labels=range(N_CLASSES), normalize='true')
    plot_confusion(cm_norm, CLASS_NAMES, f'{combo}  acc={acc:.3f}', od / 'confusion_matrix.png')
    np.savetxt(od / 'confusion_matrix_counts.csv',
               confusion_matrix(yte, pred, labels=range(N_CLASSES)),
               fmt='%d', delimiter=',', header=','.join(CLASS_NAMES), comments='')
    plot_roc(yte, prob, CLASS_NAMES, f'{combo} ROC', od / 'roc_curve.png')
    cr = classification_report(yte, pred, labels=range(N_CLASSES), target_names=CLASS_NAMES,
                               output_dict=True, zero_division=0)
    pd.DataFrame(cr).T.to_csv(od / 'per_class_metrics.csv')

    cv = pd.DataFrame(cv_rows)
    summary = dict(sensor=sensor, norm=norm, data=data, model=model,
                   cv_weighted_f1_mean=float(cv['weighted_f1'].mean()),
                   cv_weighted_f1_std=float(cv['weighted_f1'].std()),
                   cv_acc_mean=float(cv['acc'].mean()), cv_macro_f1_mean=float(cv['macro_f1'].mean()),
                   cv_auc_mean=float(cv['auc'].mean()),
                   best_cv_select=best_cv, select_metric='weighted_f1',
                   test_acc=acc, test_weighted_f1=wf1, test_macro_f1=mf1, test_auc=float(aucv),
                   best_params=best, trials=trials, duration_sec=time.time() - t0,
                   n_features=len(feat), n_train_final=int(len(Xfull)), n_test=int(len(Xte)))
    mpath.write_text(json.dumps(summary, indent=2, default=float))

    master = out_root / 'ALL_RESULTS_combined.csv'
    row = {k: summary[k] for k in ('sensor', 'norm', 'data', 'model', 'cv_weighted_f1_mean',
           'cv_weighted_f1_std', 'cv_acc_mean', 'cv_macro_f1_mean', 'cv_auc_mean',
           'test_acc', 'test_weighted_f1', 'test_macro_f1', 'test_auc', 'duration_sec')}
    row['best_params'] = json.dumps(best)
    pd.DataFrame([row]).to_csv(master, mode='a', header=not master.exists(), index=False)
    print(f'[done] {combo} {time.time() - t0:.0f}s', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sensor', choices=['sw', 'vn'], default='sw')
    p.add_argument('--norm', choices=['none', 'snv', 'msc'], required=True)
    p.add_argument('--data', choices=['raw', 'ae', 'gan'], required=True)
    p.add_argument('--model', choices=['knn', 'svm', 'rf', 'xgb', 'cnn', 'all_ml', 'all'],
                   required=True)
    p.add_argument('--ml_trials', type=int, default=200)
    p.add_argument('--cnn_trials', type=int, default=200)
    p.add_argument('--cnn_ep', type=int, default=200)
    p.add_argument('--cnn_pat', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--gpu', type=int, default=None)
    p.add_argument('--out_root', default=None, help='results root (default: repo/results)')
    p.add_argument('--splits_dir', default=None,
                   help='dir with {train,test}_<prefix>.csv (default: data/train_test_split; '
                        'point at data/per_cluster_split/<mode> for cluster-mode runs)')
    p.add_argument('--cache_dir', default=None,
                   help='augmentation cache dir (default: data/augmentation/cache_<prefix>)')
    a = p.parse_args()
    out_root = Path(a.out_root) if a.out_root else REPO / 'results'
    if a.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    models = {'all': ['knn', 'svm', 'rf', 'xgb', 'cnn'],
              'all_ml': ['knn', 'svm', 'rf', 'xgb']}.get(a.model, [a.model])
    for m in models:
        run(a.sensor, a.norm, a.data, m, a.ml_trials, a.cnn_trials, a.cnn_ep, a.cnn_pat,
            a.seed, out_root, a.splits_dir, a.cache_dir)


if __name__ == '__main__':
    main()
