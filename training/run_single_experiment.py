# -*- coding: utf-8 -*-
"""
Part A runner: one (sensor, norm, data, model) combo per invocation.

Pipeline:
  1) Load train/test CSV (raw/ae/gan), apply norm (none/snv/msc),
     attach cluster_0/1/2 features (raw: per-sample by sample_id; AE/GAN: class mean),
     fit StandardScaler on train (combining spectrum+cluster columns).
  2) Optuna search (5-fold CV on train) → best params
       --ml_trials default 100; --cnn_trials default 80
  3) 5-fold CV with best params → per-fold metrics + (CNN) training curves
  4) Fit final model on full train → evaluate on holdout test
  5) Save: confusion_matrix.png, roc_curve.png, training_curves.png (CNN),
           per_class_metrics.csv, train_cv_per_fold.csv, test_metrics.json,
           appended row in master.csv

Hyperparameter spaces (expanded):
  KNN (5):  n_neighbors, weights, p, metric, leaf_size
  SVM (5):  C, gamma, kernel, class_weight, tol
  RF  (5):  n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features
  XGB (9):  max_depth, learning_rate, n_estimators, subsample, colsample_bytree,
            reg_alpha, reg_lambda, min_child_weight, gamma
  1D-CNN:   n_filters, kernel_size, n_layers, dropout, fc_units, lr, batch_size
            + epochs=200 with early stopping (patience=10)

Splits:
  - test set is ALWAYS the holdout data/train_test_split/test_<sensor>.csv (never augmented)
  - train: data/train_test_split/train_<sensor>.csv  (raw)
           OR  data/augmentation/{ae,gan}_<sensor>_n300.csv  (augmented)
  - AE/GAN CSVs already contain (original_train + augmented) rows, no sample_id.

Usage:
  python run_single_experiment.py --sensor sw --norm none --data raw --model knn
  python run_single_experiment.py --sensor vn --norm snv --data ae --model all
  python run_single_experiment.py --sensor sw --norm none --data raw --model cnn --gpu 0
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.svm import SVC
from xgboost import XGBClassifier

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim

# normalization helper lives in ../normalization/
sys.path.insert(0, str(Path(__file__).parents[1] / 'normalization'))
from spectral_normalization import apply_msc, apply_snv

# Paths are resolved relative to this script so the repo is self-contained.
# Layout (relative to git_open/):
#   data/train_test_split/{train,test}_{swir,vnir}.csv
#   data/augmentation/{ae,gan}_{swir,vnir}_n300.csv
#   data/clustering/cluster_ratios_{swir,vnir}.csv
REPO = Path(__file__).resolve().parents[2]
DATA = REPO / 'data'
SPLIT = DATA / 'train_test_split'
AUG = DATA / 'augmentation'
CLUSTER = DATA / 'clustering'
# Per-cluster CSVs (modes c0/c1/c2/all_leaf) are produced by
# preprocessing/make_per_cluster_csvs.py and not bundled by default.
SPLIT_MODE = DATA / 'per_cluster_split'
DEFAULT_OUT = REPO / 'results' / 'single_experiment'

SENSOR_LONG = {'sw': 'swir', 'vn': 'vnir'}

CLASS_NAMES = ['DR', 'MD', 'SD', 'WW']
N_CLASSES = len(CLASS_NAMES)
CLUSTER_COLS = ['cluster_0', 'cluster_1', 'cluster_2']

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ─────────────────────────── Cluster feature attach ───────────────────────────
def get_class_mean_cluster(sensor: str) -> pd.DataFrame:
    """Class mean cluster_0/1/2 ratios computed from raw train (per-sample matched)."""
    s_long = SENSOR_LONG[sensor]
    train = pd.read_csv(SPLIT / f'train_{s_long}.csv')
    cluster = pd.read_csv(CLUSTER / f'cluster_ratios_{s_long}.csv')
    crenamed = cluster[['sample'] + CLUSTER_COLS].rename(columns={'sample': 'cluster_sample_id'})
    merged = train.merge(crenamed, on='cluster_sample_id', how='left')
    matched = merged.dropna(subset=CLUSTER_COLS)
    return matched.groupby('label')[CLUSTER_COLS].mean()


def attach_cluster_raw(df: pd.DataFrame, sensor: str, class_means: pd.DataFrame) -> pd.DataFrame:
    """For raw CSV (has cluster_sample_id): direct merge, fill missing with class mean."""
    s_long = SENSOR_LONG[sensor]
    cluster = pd.read_csv(CLUSTER / f'cluster_ratios_{s_long}.csv')
    crenamed = cluster[['sample'] + CLUSTER_COLS].rename(columns={'sample': 'cluster_sample_id'})
    merged = df.merge(crenamed, on='cluster_sample_id', how='left')
    miss = merged[CLUSTER_COLS[0]].isna()
    if miss.any():
        for col in CLUSTER_COLS:
            merged.loc[miss, col] = merged.loc[miss, 'label'].map(class_means[col])
    return merged


def attach_cluster_aug(df: pd.DataFrame, class_means: pd.DataFrame) -> pd.DataFrame:
    """For AE/GAN (no sample_id): assign class-mean cluster ratio to every row."""
    out = df.copy()
    for col in CLUSTER_COLS:
        out[col] = out['label'].map(class_means[col])
    return out


# ─────────────────────────── Data loading ───────────────────────────
def load_train_test(sensor: str, data: str, mode: str):
    """mode ∈ {all_leaf, c0, c1, c2}. c0/c1/c2 only supports data=raw."""
    if mode in ('c0', 'c1', 'c2') and data != 'raw':
        raise ValueError(f'mode={mode} only supports data=raw, got {data}')

    s_long = SENSOR_LONG[sensor]
    if mode == 'all_leaf':
        test = pd.read_csv(SPLIT_MODE / f'all_leaf_test_{s_long}.csv')
        if data == 'raw':
            train = pd.read_csv(SPLIT_MODE / f'all_leaf_train_{s_long}.csv')
        elif data == 'ae':
            train = pd.read_csv(AUG / f'ae_{s_long}_n300.csv')
        elif data == 'gan':
            train = pd.read_csv(AUG / f'gan_{s_long}_n300.csv')
        else:
            raise ValueError(data)
    else:  # c0 / c1 / c2
        train = pd.read_csv(SPLIT_MODE / f'{mode}_train_{s_long}.csv')
        test = pd.read_csv(SPLIT_MODE / f'{mode}_test_{s_long}.csv')

    feat_cols = [c for c in train.columns if c.startswith('feat_')]
    X_tr = train[feat_cols].values.astype(np.float64)
    X_te = test[feat_cols].values.astype(np.float64)
    le = LabelEncoder().fit(CLASS_NAMES)
    y_tr = le.transform(train['label'].values)
    y_te = le.transform(test['label'].values)
    return X_tr, y_tr, X_te, y_te, feat_cols, le


def normalize(X_tr, X_te, norm):
    if norm == 'none':
        Xtr, Xte = X_tr.copy(), X_te.copy()
    elif norm == 'snv':
        Xtr, Xte = apply_snv(X_tr), apply_snv(X_te)
    elif norm == 'msc':
        Xtr, ref = apply_msc(X_tr)
        Xte, _ = apply_msc(X_te, reference=ref)
    else:
        raise ValueError(norm)
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr).astype(np.float32), sc.transform(Xte).astype(np.float32)


# ─────────────────────────── ML objectives (expanded) ───────────────────────────
def ml_objective(model_name, X, y, seed):
    def obj(trial):
        if model_name == 'knn':
            m = KNeighborsClassifier(
                n_neighbors=trial.suggest_int('n_neighbors', 1, 50),
                weights=trial.suggest_categorical('weights', ['uniform', 'distance']),
                p=trial.suggest_int('p', 1, 2),
                metric=trial.suggest_categorical('metric', ['minkowski', 'manhattan', 'chebyshev']),
                leaf_size=trial.suggest_int('leaf_size', 10, 60),
            )
        elif model_name == 'svm':
            m = SVC(
                C=trial.suggest_float('C', 1e-2, 1e2, log=True),
                gamma=trial.suggest_float('gamma', 1e-4, 1.0, log=True),
                kernel=trial.suggest_categorical('kernel', ['rbf', 'linear']),
                class_weight=trial.suggest_categorical('class_weight', [None, 'balanced']),
                tol=trial.suggest_float('tol', 1e-5, 1e-2, log=True),
                probability=True, random_state=seed,
            )
        elif model_name == 'rf':
            m = RandomForestClassifier(
                n_estimators=trial.suggest_int('n_estimators', 50, 500),
                max_depth=trial.suggest_int('max_depth', 3, 30),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                random_state=seed, n_jobs=-1,
            )
        elif model_name == 'xgb':
            m = XGBClassifier(
                max_depth=trial.suggest_int('max_depth', 3, 12),
                learning_rate=trial.suggest_float('learning_rate', 1e-2, 0.3, log=True),
                n_estimators=trial.suggest_int('n_estimators', 50, 500),
                subsample=trial.suggest_float('subsample', 0.5, 1.0),
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
                reg_alpha=trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
                reg_lambda=trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
                min_child_weight=trial.suggest_float('min_child_weight', 1, 10),
                gamma=trial.suggest_float('gamma', 0, 5),
                eval_metric='mlogloss', random_state=seed, n_jobs=-1,
            )
        else:
            raise ValueError(model_name)
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        accs = []
        for tr, va in skf.split(X, y):
            m.fit(X[tr], y[tr])
            accs.append(accuracy_score(y[va], m.predict(X[va])))
        return float(np.mean(accs))
    return obj


def build_ml(model_name, params, seed):
    if model_name == 'knn':
        return KNeighborsClassifier(**params)
    if model_name == 'svm':
        return SVC(probability=True, random_state=seed, **params)
    if model_name == 'rf':
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **params)
    if model_name == 'xgb':
        return XGBClassifier(eval_metric='mlogloss', random_state=seed, n_jobs=-1, **params)
    raise ValueError(model_name)


# ─────────────────────────── 1D-CNN with early stopping ───────────────────────────
class CNN1D(nn.Module):
    def __init__(self, n_bands, n_classes, n_filters, kernel_size, n_layers, dropout, fc_units):
        super().__init__()
        layers, in_ch, cur_len = [], 1, n_bands
        for _ in range(n_layers):
            layers += [nn.Conv1d(in_ch, n_filters, kernel_size, padding=kernel_size // 2),
                       nn.ReLU(),
                       nn.MaxPool1d(2)]
            in_ch, cur_len = n_filters, cur_len // 2
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch * cur_len, fc_units), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(fc_units, n_classes),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


def cnn_train_eval(X_tr, y_tr, X_va, y_va, params, n_classes,
                   max_epochs=200, patience=10, seed=42, return_history=False):
    torch.manual_seed(seed); np.random.seed(seed)
    arch_keys = ['n_filters', 'kernel_size', 'n_layers', 'dropout', 'fc_units']
    model = CNN1D(X_tr.shape[1], n_classes, **{k: params[k] for k in arch_keys}).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=params['lr'])
    crit = nn.CrossEntropyLoss()
    Xtr = torch.from_numpy(X_tr.astype(np.float32))[:, None, :].to(DEVICE)
    ytr = torch.from_numpy(y_tr.astype(np.int64)).to(DEVICE)
    Xva = torch.from_numpy(X_va.astype(np.float32))[:, None, :].to(DEVICE)
    yva_t = torch.from_numpy(y_va.astype(np.int64)).to(DEVICE)
    bs = params['batch_size']; n = X_tr.shape[0]

    history = {'tr_loss': [], 'va_loss': [], 'va_acc': []}
    best_acc = -1.0; best_state = None; bad = 0
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        tr_loss = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = crit(model(Xtr[idx]), ytr[idx])
            loss.backward(); opt.step()
            tr_loss += loss.item() * len(idx)
        tr_loss /= n
        model.eval()
        with torch.no_grad():
            out = model(Xva)
            va_loss = crit(out, yva_t).item()
            va_pred = out.argmax(1).cpu().numpy()
            va_acc = float((va_pred == y_va).mean())
        history['tr_loss'].append(tr_loss)
        history['va_loss'].append(va_loss)
        history['va_acc'].append(va_acc)
        if va_acc > best_acc:
            best_acc = va_acc; bad = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(Xva)
        prob = torch.softmax(out, dim=1).cpu().numpy()
        pred = prob.argmax(1)
    if return_history:
        return pred, prob, history
    return pred, prob


def cnn_objective(X, y, n_classes, seed, max_epochs=200, patience=10):
    def obj(trial):
        params = {
            'n_filters': trial.suggest_categorical('n_filters', [16, 32, 64]),
            'kernel_size': trial.suggest_categorical('kernel_size', [3, 5, 7]),
            'n_layers': trial.suggest_int('n_layers', 1, 3),
            'dropout': trial.suggest_float('dropout', 0.0, 0.5),
            'fc_units': trial.suggest_categorical('fc_units', [32, 64, 128]),
            'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        }
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        accs = []
        for tr, va in skf.split(X, y):
            pred, _ = cnn_train_eval(X[tr], y[tr], X[va], y[va], params, n_classes,
                                     max_epochs=max_epochs, patience=patience, seed=seed)
            accs.append(float((pred == y[va]).mean()))
        return float(np.mean(accs))
    return obj


# ─────────────────────────── Eval helpers ───────────────────────────
def eval_metrics(y_true, y_pred, y_prob, n_classes):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    auc_v = float('nan')
    if y_prob is not None:
        try:
            yb = label_binarize(y_true, classes=range(n_classes))
            auc_v = roc_auc_score(yb, y_prob, multi_class='ovr')
        except Exception:
            pass
    return float(acc), float(f1), float(auc_v)


def plot_confusion(cm, labels, title, path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()


def plot_roc(y_true, y_prob, labels, title, path):
    n = len(labels)
    yb = label_binarize(y_true, classes=range(n))
    plt.figure(figsize=(6, 5))
    for i in range(n):
        fpr, tpr, _ = roc_curve(yb[:, i], y_prob[:, i])
        plt.plot(fpr, tpr, label=f'{labels[i]} (AUC={auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(title); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()


def plot_cnn_history(histories, path):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for fold, h in enumerate(histories):
        ep = range(1, len(h['tr_loss']) + 1)
        axs[0].plot(ep, h['tr_loss'], label=f'fold{fold}')
        axs[1].plot(ep, h['va_loss'], label=f'fold{fold}')
        axs[2].plot(ep, h['va_acc'], label=f'fold{fold}')
    axs[0].set_title('train loss'); axs[1].set_title('val loss'); axs[2].set_title('val acc')
    for a in axs:
        a.set_xlabel('epoch'); a.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


# ─────────────────────────── Per-combo runner ───────────────────────────
def run_combo(sensor, norm, data, mode, model, ml_trials, cnn_trials, seed, out_root,
              cnn_max_epochs, cnn_patience):
    combo_id = f'{sensor}_{mode}_{norm}_{data}_{model}'
    out_dir = Path(out_root) / 'per_combo' / combo_id
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / 'test_metrics.json'
    if metrics_path.exists():
        print(f'[skip] {combo_id} already done')
        return

    trials = cnn_trials if model == 'cnn' else ml_trials
    print(f'\n=== {combo_id} (trials={trials}) ===')
    t0 = time.time()
    X_tr_raw, y_tr, X_te_raw, y_te, feat_cols, le = load_train_test(sensor, data, mode)
    print(f'  train n={len(X_tr_raw)}, test n={len(X_te_raw)}, n_features={X_tr_raw.shape[1]}')

    X_tr, X_te = normalize(X_tr_raw, X_te_raw, norm)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage_url = f'sqlite:///{Path(out_root)}/optuna_studies.db'
    Path(out_root).mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=seed),
                                storage=storage_url,
                                study_name=combo_id,
                                load_if_exists=True)
    if model == 'cnn':
        study.optimize(cnn_objective(X_tr, y_tr, N_CLASSES, seed,
                                     max_epochs=cnn_max_epochs, patience=cnn_patience),
                       n_trials=trials, show_progress_bar=False)
    else:
        study.optimize(ml_objective(model, X_tr, y_tr, seed),
                       n_trials=trials, show_progress_bar=False)
    best_params = dict(study.best_params)
    best_cv = float(study.best_value)
    print(f'  best CV acc (Optuna): {best_cv:.4f}')

    # Save full Optuna trial history + visualization PNGs
    trials_df = study.trials_dataframe()
    trials_df.to_csv(out_dir / 'optuna_trials.csv', index=False)
    try:
        from optuna.visualization.matplotlib import (
            plot_optimization_history,
            plot_param_importances,
        )
        plot_optimization_history(study); plt.tight_layout()
        plt.savefig(out_dir / 'optuna_history.png', dpi=150); plt.close()
        plot_param_importances(study); plt.tight_layout()
        plt.savefig(out_dir / 'optuna_param_importance.png', dpi=150); plt.close()
    except Exception as e:
        print(f'  (optuna viz skipped: {e})')

    # 5-fold CV with best params
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    cv_rows, cnn_histories = [], []
    for fold, (tri, vai) in enumerate(skf.split(X_tr, y_tr)):
        if model == 'cnn':
            pred, prob, hist = cnn_train_eval(
                X_tr[tri], y_tr[tri], X_tr[vai], y_tr[vai],
                best_params, N_CLASSES,
                max_epochs=cnn_max_epochs, patience=cnn_patience,
                seed=seed, return_history=True)
            cnn_histories.append(hist)
        else:
            mm = build_ml(model, best_params, seed)
            mm.fit(X_tr[tri], y_tr[tri])
            pred = mm.predict(X_tr[vai])
            prob = mm.predict_proba(X_tr[vai])
        a, f, au = eval_metrics(y_tr[vai], pred, prob, N_CLASSES)
        cv_rows.append({'fold': fold, 'acc': a, 'f1': f, 'auc': au})
    pd.DataFrame(cv_rows).to_csv(out_dir / 'train_cv_per_fold.csv', index=False)
    if cnn_histories:
        plot_cnn_history(cnn_histories, out_dir / 'training_curves.png')

    # Final fit on full train, evaluate on holdout test
    if model == 'cnn':
        pred_te, prob_te, hist_te = cnn_train_eval(
            X_tr, y_tr, X_te, y_te, best_params, N_CLASSES,
            max_epochs=cnn_max_epochs, patience=cnn_patience,
            seed=seed, return_history=True)
        plot_cnn_history([hist_te], out_dir / 'training_curves_final.png')
    else:
        mm = build_ml(model, best_params, seed)
        mm.fit(X_tr, y_tr)
        pred_te = mm.predict(X_te)
        prob_te = mm.predict_proba(X_te)
    acc, f1, aucv = eval_metrics(y_te, pred_te, prob_te, N_CLASSES)
    print(f'  test acc={acc:.4f} f1={f1:.4f} auc={aucv:.4f}')

    cm = confusion_matrix(y_te, pred_te, normalize='true')
    plot_confusion(cm, le.classes_, f'{combo_id} test_acc={acc:.3f}',
                   out_dir / 'confusion_matrix.png')
    plot_roc(y_te, prob_te, le.classes_, f'{combo_id} ROC', out_dir / 'roc_curve.png')
    cr = classification_report(y_te, pred_te, target_names=le.classes_,
                               output_dict=True, zero_division=0)
    pd.DataFrame(cr).T.to_csv(out_dir / 'per_class_metrics.csv')

    cv = pd.DataFrame(cv_rows)
    summary = {
        'sensor': sensor, 'mode': mode, 'norm': norm, 'data': data, 'model': model,
        'cv_acc_mean': float(cv['acc'].mean()), 'cv_acc_std': float(cv['acc'].std()),
        'cv_f1_mean': float(cv['f1'].mean()), 'cv_auc_mean': float(cv['auc'].mean()),
        'test_acc': acc, 'test_f1': f1, 'test_auc': aucv,
        'best_cv_acc': best_cv, 'best_params': best_params,
        'trials': trials, 'duration_sec': time.time() - t0,
        'n_features': int(X_tr.shape[1]), 'n_train': int(len(X_tr)), 'n_test': int(len(X_te)),
    }
    metrics_path.write_text(json.dumps(summary, indent=2, default=float))

    master = Path(out_root) / 'master.csv'
    master.parent.mkdir(parents=True, exist_ok=True)
    row = {k: summary[k] for k in
           ('sensor', 'mode', 'norm', 'data', 'model',
            'cv_acc_mean', 'cv_f1_mean', 'cv_auc_mean',
            'test_acc', 'test_f1', 'test_auc', 'duration_sec')}
    row['best_params'] = json.dumps(best_params)
    pd.DataFrame([row]).to_csv(master, mode='a', header=not master.exists(), index=False)

    print(f'[done] {combo_id} {time.time()-t0:.0f}s')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sensor', choices=['sw', 'vn'], required=True)
    p.add_argument('--mode', choices=['all_leaf', 'c0', 'c1', 'c2'], required=True)
    p.add_argument('--norm', choices=['none', 'snv', 'msc'], required=True)
    p.add_argument('--data', choices=['raw', 'ae', 'gan'], required=True)
    p.add_argument('--model', choices=['knn', 'svm', 'rf', 'xgb', 'cnn', 'all_ml', 'all'],
                   required=True)
    p.add_argument('--ml_trials', type=int, default=100)
    p.add_argument('--cnn_trials', type=int, default=80)
    p.add_argument('--cnn_max_epochs', type=int, default=200)
    p.add_argument('--cnn_patience', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out_root', default=DEFAULT_OUT)
    p.add_argument('--gpu', type=int, default=None,
                   help='If set, restrict CUDA to this GPU index (for parallel runs).')
    args = p.parse_args()

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    if args.model == 'all':
        models = ['knn', 'svm', 'rf', 'xgb', 'cnn']
    elif args.model == 'all_ml':
        models = ['knn', 'svm', 'rf', 'xgb']
    else:
        models = [args.model]
    for m in models:
        run_combo(args.sensor, args.norm, args.data, args.mode, m,
                  args.ml_trials, args.cnn_trials, args.seed, args.out_root,
                  args.cnn_max_epochs, args.cnn_patience)


if __name__ == '__main__':
    main()
