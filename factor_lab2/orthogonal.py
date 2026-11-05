# -*- coding: utf-8 -*-
"""
因子正交与降维：Gram-Schmidt、对称正交、残差正交、PCA。
输入 F 为因子矩阵（样本 × 因子，已对齐、去 NaN）。
"""
import numpy as np
import pandas as pd


def _matrix_inverse_sqrt(M):
    """对称正定矩阵 M 的逆平方根 M^(-1/2)。"""
    vals, vecs = np.linalg.eigh(M)
    vals = np.clip(vals, 1e-12, None)
    return vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T


def _center(X):
    """列去均值（因子正交前先中心化，保证正交后相关也为0）。"""
    return X - X.mean(axis=0, keepdims=True)


def schmidt_orthogonal(F):
    """Gram-Schmidt 正交化（结果依赖因子顺序），先列中心化。"""
    X = _center(np.asarray(F, float))
    Q = np.zeros_like(X)
    for j in range(X.shape[1]):
        v = X[:, j].copy()
        for k in range(j):
            v -= (Q[:, k] @ v) / (Q[:, k] @ Q[:, k]) * Q[:, k]
        Q[:, j] = v
    return pd.DataFrame(Q, index=F.index, columns=F.columns)


def symmetric_orthogonal(F):
    """对称正交：F_orth = F (F^T F)^(-1/2)，先列中心化，不依赖顺序、信息保留均衡。"""
    X = _center(np.asarray(F, float))
    M = X.T @ X
    Q = X @ _matrix_inverse_sqrt(M)
    return pd.DataFrame(Q, index=F.index, columns=F.columns)


def residual_orthogonal(target, base):
    """target 对 base（可多个）回归取残差，剔除与 base 重叠的信息。"""
    y = np.asarray(target, float)
    B = np.column_stack([np.ones(len(y)), np.asarray(base, float)])
    beta, *_ = np.linalg.lstsq(B, y, rcond=None)
    return pd.Series(y - B @ beta, index=target.index)


def pca_factors(F, n_components=None, standardize=True):
    """PCA 降维，返回主成分、解释方差比、载荷。"""
    X = np.asarray(F, float)
    if standardize:
        X = (X - X.mean(0)) / np.where(X.std(0) == 0, np.nan, X.std(0))
    cov = np.cov(X, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    if n_components is None:
        n_components = len(vals)
    comps = X @ vecs[:, :n_components]
    var_ratio = vals[:n_components] / vals.sum()
    return {
        "components": pd.DataFrame(comps, index=F.index,
                                   columns=["PC%d" % (i+1) for i in range(n_components)]),
        "explained_variance_ratio": var_ratio,
        "loadings": pd.DataFrame(vecs[:, :n_components], index=F.columns,
                                 columns=["PC%d" % (i+1) for i in range(n_components)]),
    }
