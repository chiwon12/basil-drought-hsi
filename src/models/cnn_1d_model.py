# -*- coding: utf-8 -*-
"""1D-CNN classifier — paper architecture (PaperCNN, PyTorch).

This is the manuscript's 1D-CNN: a faithful PyTorch port of the Keras
`build_model` (SAT_BASIL/src/model/1d_cnn.py). Two fixed conv blocks, each:

    Conv1d(valid padding) -> ReLU -> [BatchNorm?] -> MaxPool -> Dropout

then Flatten or GlobalAveragePooling1D -> Dense(ReLU) -> Dense(n_classes).

Hybrid hyper-parameter space (per block where noted):
  * structural (from 1d_cnn.py): batch_norm{T,F}, pool_size{2,4}, gap_type{flatten,global}
  * value ranges (paper Table S1): filters{16,32,64}, kernel{3,5,7}, dropout 0–0.5,
    dense{64,128}, lr 1e-4–1e-2, batch_size{16,32,64}

The data/CV/leakage protocol is identical to the classical models (the training
driver reuses the same fold-aware loaders + per-fold normalization). This module
exposes the same interface so `training/run_single_experiment.py` works unchanged:
  * `suggest_cnn(trial)`  — Optuna search space;
  * `cnn_train_eval(...)` — CV-only: early stop on the fold's val loss, restore the
    best-loss weights, return validation predictions + history;
  * `cnn_fit_fixed(...)`  — final model only: train for a FIXED epoch count (the
    mean CV best-epoch) without touching the test set, then predict.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# Determinism so seed=42 reproduces (avoids cuDNN nondeterministic jitter).
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class PaperCNN(nn.Module):
    """Keras build_model parity. Conv1d default padding='valid' (length shrinks).
    Block order: Conv(ReLU) -> (BatchNorm) -> MaxPool -> Dropout."""

    def __init__(self, n_bands, n_classes, p):
        super().__init__()

        def block(in_ch, f, k, bn, pool, drop):
            ls = [nn.Conv1d(in_ch, f, k), nn.ReLU()]
            if bn:
                ls.append(nn.BatchNorm1d(f))
            ls += [nn.MaxPool1d(pool), nn.Dropout(drop)]
            return ls, f

        l1, c1 = block(1, p['filters1'], p['kernel_size1'], p['batch_norm1'],
                       p['pool_size1'], p['dropout1'])
        l2, c2 = block(c1, p['filters2'], p['kernel_size2'], p['batch_norm2'],
                       p['pool_size2'], p['dropout2'])
        self.conv = nn.Sequential(*l1, *l2)
        self.gap = (p['gap_type'] == 'global')
        with torch.no_grad():                       # flatten dim depends on valid-conv length
            L = self.conv(torch.zeros(1, 1, n_bands)).shape[-1]
        head_in = c2 if self.gap else c2 * L
        self.head = nn.Sequential(nn.Linear(head_in, p['dense']), nn.ReLU(),
                                  nn.Linear(p['dense'], n_classes))

    def forward(self, x):
        x = self.conv(x)
        x = x.mean(-1) if self.gap else x.flatten(1)   # GlobalAveragePooling1D | Flatten
        return self.head(x)


def suggest_cnn(t):
    """Hybrid HP space: structural from 1d_cnn.py, value ranges from Table S1.
    filters/kernel/pool/dropout/batch_norm are per block (1, 2)."""
    p = {}
    for b in (1, 2):
        p[f'filters{b}'] = t.suggest_categorical(f'filters{b}', [16, 32, 64])
        p[f'kernel_size{b}'] = t.suggest_categorical(f'kernel_size{b}', [3, 5, 7])
        p[f'batch_norm{b}'] = t.suggest_categorical(f'batch_norm{b}', [True, False])
        p[f'pool_size{b}'] = t.suggest_categorical(f'pool_size{b}', [2, 4])
        p[f'dropout{b}'] = t.suggest_float(f'dropout{b}', 0.0, 0.5)
    p['gap_type'] = t.suggest_categorical('gap_type', ['flatten', 'global'])
    p['dense'] = t.suggest_categorical('dense', [64, 128])
    p['lr'] = t.suggest_float('lr', 1e-4, 1e-2, log=True)
    p['batch_size'] = t.suggest_categorical('batch_size', [16, 32, 64])
    return p


def cnn_train_eval(Xtr, ytr, Xva, yva, p, n_classes, max_epochs=200, patience=10,
                   seed=42, return_history=False):
    """CV-only: early-stop on val loss, restore best-loss weights, predict on val."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = PaperCNN(Xtr.shape[1], n_classes, p).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=p['lr']); crit = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(Xtr.astype(np.float32))[:, None, :].to(DEVICE)
    yt = torch.from_numpy(ytr.astype(np.int64)).to(DEVICE)
    Xv = torch.from_numpy(Xva.astype(np.float32))[:, None, :].to(DEVICE)
    yv = torch.from_numpy(yva.astype(np.int64)).to(DEVICE)
    bs, n = p['batch_size'], len(Xtr)
    hist = {'tr_loss': [], 'va_loss': [], 'va_acc': []}
    best_loss, best_state, bad = float('inf'), None, 0
    for _ in range(max_epochs):
        model.train(); perm = torch.randperm(n, device=DEVICE); tl = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if idx.numel() < 2:               # BatchNorm: variance undefined for batch of 1 -> skip
                continue
            opt.zero_grad()
            loss = crit(model(Xt[idx]), yt[idx]); loss.backward(); opt.step()
            tl += loss.item() * idx.numel()
        tl /= n
        model.eval()
        with torch.no_grad():
            out = model(Xv); vl = crit(out, yv).item()
            va_acc = float((out.argmax(1).cpu().numpy() == yva).mean())
        hist['tr_loss'].append(tl); hist['va_loss'].append(vl); hist['va_acc'].append(va_acc)
        if vl < best_loss - 1e-5:
            best_loss, bad = vl, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(Xv); prob = torch.softmax(out, 1).cpu().numpy(); pred = prob.argmax(1)
    return (pred, prob, hist) if return_history else (pred, prob)


def cnn_fit_fixed(Xtr, ytr, p, n_classes, n_epochs, seed, Xpred):
    """Final model only: train for a FIXED n_epochs (no early stopping, test never
    inspected), then predict on Xpred."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = PaperCNN(Xtr.shape[1], n_classes, p).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=p['lr']); crit = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(Xtr.astype(np.float32))[:, None, :].to(DEVICE)
    yt = torch.from_numpy(ytr.astype(np.int64)).to(DEVICE)
    bs, n = p['batch_size'], len(Xtr)
    for _ in range(max(1, n_epochs)):
        model.train(); perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if idx.numel() < 2:
                continue
            opt.zero_grad()
            crit(model(Xt[idx]), yt[idx]).backward(); opt.step()
    model.eval()
    Xp = torch.from_numpy(Xpred.astype(np.float32))[:, None, :].to(DEVICE)
    with torch.no_grad():
        out = model(Xp); prob = torch.softmax(out, 1).cpu().numpy(); pred = prob.argmax(1)
    return pred, prob
