# -*- coding: utf-8 -*-
"""
WorldQuant Alpha101 / 国信 Alpha191 精选复现（日频 OHLCV）。
只实现有代表性的若干个，标注原始编号；完整批量实现可按同一算子库扩展。
"""
import numpy as np


# ---------- 算子 ----------
def cs_rank(x):
    return x.rank(axis=1, pct=True)


def delta(x, d):
    return x.diff(d)


def delay(x, d):
    return x.shift(d)


def ts_corr(a, b, d):
    return a.rolling(d).corr(b)


def ts_rank(x, d):
    return x.rolling(d).rank(pct=True)


def signed_power(x, e):
    return np.sign(x) * np.abs(x) ** e


def build_alphas(open_, close, volume, high, low):
    a = {}

    # Alpha191#1：-1 × ts_corr(rank(open), rank(volume), 10)
    a["Alpha191#1"] = -ts_corr(cs_rank(open_), cs_rank(volume), 10)

    # Alpha101#6：-1 × ts_corr(open, volume, 10)
    a["Alpha101#6"] = -ts_corr(open_, volume, 10)

    # Alpha101#12：sign(delta(volume,1)) × (-delta(close,1))
    a["Alpha101#12"] = np.sign(delta(volume, 1)) * (-delta(close, 1))

    # Alpha191#3 简化：-ts_rank(corr(close,volume,2), 10)
    a["Alpha191#3"] = -ts_rank(ts_corr(close, volume, 2), 10)

    # 反转类：-delta(close,5) 配合 rank
    a["Alpha101 反转变体"] = -cs_rank(delta(close, 5))

    # 振幅反转
    amplitude = (high - low) / close
    a["Alpha191 振幅反转"] = -ts_rank(amplitude, 20)
    return a
