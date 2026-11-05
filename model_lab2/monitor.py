# -*- coding: utf-8 -*-
"""模型监控与衰减：滚动性能、PSI 分布漂移、特征漂移、衰减预警。"""
import numpy as np
import pandas as pd


def rolling_ic_by_date(pred_df, label_col="y", pred_col="pred", window=60):
    """按日期的 IC 序列及其滚动均值。"""
    ics = []
    for dt, sub in pred_df.groupby(level=0):
        if len(sub) > 10:
            ics.append((dt, sub[pred_col].corr(sub[label_col],
                                                method="spearman")))
    s = pd.Series(dict(ics))
    return s, s.rolling(window).mean()


def psi(expected, actual, bins=10):
    """总体稳定性指标 PSI：用基准期分箱边界，比较两期分布占比。

    PSI<0.1 稳定；0.1~0.25 轻微漂移；>0.25 显著漂移。
    """
    expected = np.asarray(expected, float)
    actual = np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    pe = np.histogram(expected, edges)[0] / len(expected)
    pa = np.histogram(actual, edges)[0] / len(actual)
    pe = np.where(pe == 0, 1e-6, pe)
    pa = np.where(pa == 0, 1e-6, pa)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def feature_drift(train_X, new_X):
    """逐特征的均值/标准差漂移（标准化差异）。"""
    train_X = pd.DataFrame(train_X)
    new_X = pd.DataFrame(new_X)
    rows = {}
    for c in train_X.columns:
        sd = train_X[c].std()
        rows[c] = {
            "mean_shift": (new_X[c].mean() - train_X[c].mean()) / sd,
            "std_ratio": new_X[c].std() / sd,
        }
    return pd.DataFrame(rows).T


def decay_alert(ic_series, threshold=0.0, n_below=3):
    """连续 n_below 期 IC 低于阈值则触发衰减预警。"""
    vals = np.asarray(ic_series, float)
    below = vals < threshold
    alerts = []
    run = 0
    for i, b in enumerate(below):
        run = run + 1 if b else 0
        if run >= n_below:
            alerts.append(i)
    return {"alerts": alerts, "n_alert": len(alerts),
            "latest_ic": vals[-1] if len(vals) else np.nan}
