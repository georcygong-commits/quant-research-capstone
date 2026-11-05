# -*- coding: utf-8 -*-
"""标签工程：多种监督目标的构造。"""
import numpy as np
import pandas as pd


def fwd_return_label(close, horizon):
    """固定期限未来收益（小数）。"""
    return close.shift(-horizon) / close - 1


def cs_rank_label(close, horizon):
    """截面 rank 标签到 [-1,1]，与排序目标一致、抗异常值。"""
    r = fwd_return_label(close, horizon)
    return r.rank(axis=1, pct=True) * 2 - 1


def cs_zscore_label(close, horizon, winsorize=3.0):
    """截面 zscore 标签（先去极值）。"""
    r = fwd_return_label(close, horizon)
    z = r.sub(r.mean(axis=1), axis=0).div(r.std(axis=1), axis=0)
    return z.clip(-winsorize, winsorize)


def quantile_label(close, horizon, n_quantile=5):
    """截面分位类别标签（0~n-1），用于分类。"""
    r = fwd_return_label(close, horizon)
    return r.apply(lambda s: pd.qcut(s, n_quantile, labels=False,
                                     duplicates="drop"), axis=1)


def binary_label(close, horizon):
    """二分类标签：未来收益为正=1。"""
    r = fwd_return_label(close, horizon)
    return (r > 0).astype(float)


def risk_adjusted_label(close, horizon, vol_window=20):
    """风险调整标签：未来收益 / 当期波动。"""
    r = fwd_return_label(close, horizon)
    vol = close.pct_change().rolling(vol_window).std()
    return r / vol


def multi_horizon_labels(close, horizons=(1,5,20)):
    """多周期标签字典。"""
    return {h: cs_rank_label(close, h) for h in horizons}
