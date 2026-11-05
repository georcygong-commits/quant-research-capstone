# -*- coding: utf-8 -*-
"""
因子评估：未来收益、IC / RankIC、IR、IC 胜率、IC 衰减。
"""
import numpy as np
import pandas as pd


def fwd_returns(close, k):
    """未来 k 日收益（t 收盘 -> t+k 收盘）：close.shift(-k)/close - 1。"""
    return close.shift(-k) / close - 1


def ic_series(factor, fwd, method="spearman", min_count=10):
    """
    逐截面计算因子与未来收益的相关系数。
    method='spearman' -> RankIC（A股常用，稳健）；'pearson' -> IC。
    """
    out = {}
    common = factor.index.intersection(fwd.index)
    for dt in common:
        f, r = factor.loc[dt], fwd.loc[dt]
        v = f.notna() & r.notna()
        if v.sum() > min_count:
            out[dt] = f[v].corr(r[v], method=method)
    return pd.Series(out).sort_index()


def ic_summary(ic):
    """IC 统计：均值、标准差、IR、胜率。"""
    return {
        "IC均值": ic.mean(),
        "IC标准差": ic.std(),
        "IR": ic.mean() / ic.std() if ic.std() > 0 else np.nan,
        "IC胜率": (ic > 0).mean(),
    }


def ic_decay(factor, close, ks=(1, 2, 3, 5, 10, 20), method="spearman"):
    """对不同持有期 k 计算 IC，返回 DataFrame（含 IC均值、IR、胜率）。"""
    rows = []
    for k in ks:
        ic = ic_series(factor, fwd_returns(close, k), method=method)
        s = ic_summary(ic)
        s["持有期"] = k
        rows.append(s)
    return pd.DataFrame(rows).set_index("持有期")[["IC均值", "IR", "IC胜率"]]
