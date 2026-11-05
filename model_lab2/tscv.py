# -*- coding: utf-8 -*-
"""时间序列交叉验证与超参优化（自实现，替代 optuna）。"""
import itertools
import numpy as np
import pandas as pd


def walk_forward_splits(n, n_splits=5, train_frac=0.5):
    """扩展窗口 walk-forward：训练起点固定 0，终点逐步后移，测试紧随其后。"""
    first_train = int(n * train_frac)
    test_size = max(1, (n - first_train) // n_splits)
    splits = []
    for k in range(n_splits):
        tr_end = first_train + k * test_size
        te_end = min(tr_end + test_size, n)
        if tr_end >= n:
            break
        splits.append((np.arange(0, tr_end), np.arange(tr_end, te_end)))
    return splits


def rolling_splits(n, train_window, test_window):
    """固定窗口滚动：训练窗长度不变，整体向前滑动。"""
    splits = []
    start = 0
    while start + train_window + test_window <= n:
        tr_end = start + train_window
        splits.append((np.arange(start, tr_end),
                       np.arange(tr_end, tr_end + test_window)))
        start += test_window
    return splits


def _iter_grid(param_grid):
    keys = list(param_grid)
    for combo in itertools.product(*[param_grid[k] for k in keys]):
        yield dict(zip(keys, combo))


def default_rank_score(y_true, y_pred):
    return pd.Series(y_pred).corr(pd.Series(np.asarray(y_true)),
                                  method="spearman")


def grid_search_ts(make_model, param_grid, X, y, splits, score_fn=None):
    """在时间序列切分上做网格搜索（score 越大越好），返回最佳与全部结果。"""
    if score_fn is None:
        score_fn = default_rank_score
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    rows = []
    for params in _iter_grid(param_grid):
        fold_scores = []
        for tr_idx, te_idx in splits:
            model = make_model(**params)
            model.fit(X[tr_idx], y[tr_idx])
            pred = model.predict(X[te_idx])
            fold_scores.append(score_fn(y[te_idx], pred))
        rows.append({"params": params, "mean_score": np.mean(fold_scores),
                     "fold_scores": fold_scores})
    best = max(rows, key=lambda r: r["mean_score"])
    return best, rows
