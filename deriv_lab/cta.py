# -*- coding: utf-8 -*-
"""
CTA/管理期货（M7.1）：
时序动量、均线趋势过滤、基差与展期收益（期货期限结构）。
"""
import numpy as np


def ts_momentum(prices, lookback):
    """时序动量信号：过去 lookback 收益为正做多、为负做空。"""
    prices = np.asarray(prices, dtype=float)
    ret = prices[lookback:] / prices[:-lookback] - 1.0
    signals = np.sign(ret)
    return np.concatenate([np.full(lookback, np.nan), signals])


def ma_trend(prices, fast, slow):
    """均线趋势：快线上穿慢线做多、下穿做空（教学用简单移动平均）。"""
    p = np.asarray(prices, dtype=float)
    def ma(w):
        k = np.ones(w) / w
        out = np.full(len(p), np.nan)
        out[w-1:] = np.convolve(p, k, mode="valid")
        return out
    diff = ma(fast) - ma(slow)
    return np.where(np.isnan(diff), np.nan, np.sign(diff))


def roll_yield(near_px, far_px, days_between):
    """
    展期收益：近月相对远月的年化收益。
    近月>远月（贴水 backwardation）展期收益为正，是 CTA 常见收益来源；
    近月<远月（升水 contango）展期收益为负。
    """
    annual = (near_px / far_px) ** (365.0 / days_between) - 1.0
    return float(annual)


def basis(spot, futures):
    """基差率 = 现货/期货 - 1。"""
    return float(spot / futures - 1.0)
