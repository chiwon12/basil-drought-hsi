# -*- coding: utf-8 -*-
"""SNV / MSC implementations (per-row SNV; MSC fits each spectrum to a reference)."""
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
