# -*- coding: utf-8 -*-
"""特征选择与稳定性。"""
import numpy as np
import pandas as pd


def missing_rate(feature):
    """截面缺失率（逐期平均）。"""
    return feature.isna().mean(axis=1).mean()


def ic_filter(features, label, ic_threshold=0.02):
    """按 |IC| 均值筛选特征。"""
    keep = []
    for n, f in features.items():
        ic = _mean_rank_ic(f, label)
        if abs(ic) >= ic_threshold:
            keep.append((n, ic))
    return dict(keep)


def correlation_filter(features, corr_threshold=0.9):
    """贪心去冗余：依次保留与已选特征相关性低于阈值的特征。"""
    names = list(features.keys())
    # 用截面均值序列的相关代表特征共动
    means = pd.DataFrame({n: f.mean(axis=1) for n, f in features.items()})
    corr = means.corr()
    selected = []
    for n in names:
        if all(abs(corr.loc[n, s]) < corr_threshold for s in selected):
            selected.append(n)
    return {n: features[n] for n in selected}


def stability_score(ic_series):
    """特征稳定性：IC 均值/IC 标准差（类似 IR）。"""
    return ic_series.mean() / ic_series.std()


def _mean_rank_ic(f, label):
    ics = []
    for dt in f.index:
        a, b = f.loc[dt], label.loc[dt]
        ok = a.notna() & b.notna()
        if ok.sum() > 10:
            ics.append(a[ok].corr(b[ok], method="spearman"))
    return np.mean(ics) if ics else 0.0
