# -*- coding: utf-8 -*-
"""SNV / MSC implementations (per-row SNV; MSC fits each spectrum to a reference).

`normalize_train_test` is the leakage-safe wrapper used by the v2 CV loop: every
quantity that is *learnt* (the MSC reference spectrum and the StandardScaler
statistics) is fit on the training part only and then applied to the held-out
part. SNV is per-row so it carries no cross-sample state.
"""
from sklearn.preprocessing import StandardScaler
import numpy as np


def apply_snv(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd == 0] = 1e-10
    return (X - mu) / sd


def apply_msc(X: np.ndarray, reference: np.ndarray | None = None):
    """Returns (X_msc, reference). If reference is None, uses mean of input."""
    X = np.asarray(X, dtype=np.float64)
    if reference is None:
        reference = X.mean(axis=0)
    out = np.zeros_like(X)
    for i in range(X.shape[0]):
        a, b = np.polyfit(reference, X[i], 1)
        if abs(a) < 1e-10:
            out[i] = X[i] - b
        else:
            out[i] = (X[i] - b) / a
    return out, reference


def normalize_train_test(Xtr: np.ndarray, Xte: np.ndarray, norm: str):
    """Leakage-safe normalization. norm in {'none','snv','msc'}.

    Fits everything learnable on Xtr only, transforms both. Returns float32
    (Xtr_out, Xte_out). Used per CV fold and for the final train/test fit.
    """
    if norm == 'none':
        Xtr2, Xte2 = np.asarray(Xtr, np.float64), np.asarray(Xte, np.float64)
    elif norm == 'snv':
        Xtr2, Xte2 = apply_snv(Xtr), apply_snv(Xte)
    elif norm == 'msc':
        Xtr2, ref = apply_msc(Xtr)
        Xte2, _ = apply_msc(Xte, reference=ref)
    else:
        raise ValueError(f'unknown norm: {norm}')
    sc = StandardScaler().fit(Xtr2)
    return sc.transform(Xtr2).astype(np.float32), sc.transform(Xte2).astype(np.float32)
