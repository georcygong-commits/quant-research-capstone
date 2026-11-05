# -*- coding: utf-8 -*-
"""模型可解释性（自实现，替代 shap）：特征重要性、排列重要性、部分依赖。"""
import numpy as np
import pandas as pd


def model_feature_importance(model, feature_names):
    """树模型自带的分裂重要性。"""
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return pd.Series(dtype=float)
    return pd.Series(imp, index=feature_names).sort_values(ascending=False)


def permutation_importance(model, X, y, score_fn, n_repeat=3, seed=0):
    """
    排列重要性：逐列打乱，比较打分下降幅度。
    score_fn(model, X, y) -> 越大越好的标量（如 IC）。
    """
    rng = np.random.default_rng(seed)
    base = score_fn(model, X, y)
    out = {}
    for col in X.columns:
        drops = []
        for _ in range(n_repeat):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].values)
            drops.append(base - score_fn(model, Xp, y))
        out[col] = np.mean(drops)
    return pd.Series(out).sort_values(ascending=False)


def partial_dependence(model, X, feature, grid=None):
    """
    部分依赖：把该特征设为网格各值，取平均预测，刻画边际效应。
    """
    if grid is None:
        grid = np.quantile(X[feature].dropna(), np.linspace(0.05, 0.95, 12))
    preds = []
    for g in grid:
        Xg = X.copy()
        Xg[feature] = g
        preds.append(np.mean(model.predict(Xg)))
    return pd.Series(preds, index=grid)
