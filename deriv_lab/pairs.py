# -*- coding: utf-8 -*-
"""
配对/统计套利（M7.4）：
Engle-Granger 协整检验、对冲比、价差 z-score、均值回归信号与半衰期。
"""
import numpy as np
from statsmodels.tsa.stattools import adfuller


def hedge_ratio(y, x):
    """OLS 对冲比（Engle-Granger 第一步）。"""
    x = np.asarray(x); y = np.asarray(y)
    return float(np.polyfit(x, y, 1)[0])


def spread(y, x, beta):
    """协整价差 = y - beta*x。"""
    return np.asarray(y) - beta * np.asarray(x)


def zscore(s):
    """标准化价差（全样本口径，实盘用滚动窗口）。"""
    s = np.asarray(s)
    return (s - s.mean()) / s.std()


def coint_adf(s):
    """对价差做 ADF 单位根检验，返回 (统计量, p值)；p 小则拒绝单位根、支持协整。"""
    res = adfuller(np.asarray(s), result_object=False)
    return float(res[0]), float(res[1])


def pairs_signal(z, entry=2.0, exit=0.5):
    """z 超阈值开仓（价差回归方向），回到 exit 内平仓。"""
    sig = np.zeros_like(z)
    pos = 0
    for i, val in enumerate(z):
        if pos == 0:
            if val > entry: pos = -1
            elif val < -entry: pos = 1
        elif abs(val) < exit:
            pos = 0
        sig[i] = pos
    return sig


def half_life(s):
    """均值回归半衰期：对 Δs 关于滞后 s 做回归，半衰期 = -ln2/斜率。"""
    s = np.asarray(s)
    lag = s[:-1]; delta = np.diff(s)
    beta = np.polyfit(lag, delta, 1)[0]
    return float(-np.log(2) / beta) if beta < 0 else np.inf
