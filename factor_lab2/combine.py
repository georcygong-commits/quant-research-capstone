# -*- coding: utf-8 -*-
"""
因子合成与择时：截面标准化、等权、IC 加权、ICIR 最大化、滚动择时。
因子为宽表（date × code）；IC 历史为 DataFrame（date × factor）。
"""
import numpy as np
import pandas as pd


def cross_zscore(df):
    """截面 z-score 标准化。"""
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)


def equal_combine(factors):
    """等权合成（先截面标准化）。"""
    z = [cross_zscore(f) for f in factors]
    return sum(z) / len(z)


def ic_weight_combine(factors, ic_means):
    """按历史 IC 均值加权（方向已含在 IC 符号中）。"""
    z = [cross_zscore(f) for f in factors]
    w = np.asarray(ic_means, float)
    w = w / np.abs(w).sum()
    out = sum(wi * fi for wi, fi in zip(w, z))
    return out


def icir_weights(ic_history):
    """
    最大化合成 ICIR 的权重：w ∝ Σ_IC^-1 μ_IC，再归一化。
    """
    mu = ic_history.mean().values
    cov = ic_history.cov().values
    inv = np.linalg.pinv(cov)
    raw = inv @ mu
    w = raw / np.abs(raw).sum()
    return pd.Series(w, index=ic_history.columns)


def icir_combine(factors, ic_history):
    z = [cross_zscore(f) for f in factors]
    w = icir_weights(ic_history).values
    return sum(wi * fi for wi, fi in zip(w, z))


def timing_weights(ic_history, window=60):
    """
    因子择时：每个时点用过去 window 的 IC 计算 ICIR 权重，
    返回权重 DataFrame（date × factor），近期表现好的因子权重更高。
    """
    rows = {}
    cols = ic_history.columns
    for dt in ic_history.index:
        win = ic_history.loc[:dt].tail(window)
        if len(win) < 10:
            rows[dt] = pd.Series(1.0/len(cols), index=cols)
            continue
        mu = win.mean().values
        cov = win.cov().values
        raw = np.linalg.pinv(cov) @ mu
        w = raw / np.abs(raw).sum()
        rows[dt] = pd.Series(w, index=cols)
    return pd.DataFrame(rows).T
