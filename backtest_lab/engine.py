# -*- coding: utf-8 -*-
"""向量化回测引擎：目标权重 -> 组合收益/净值/换手（含成本）。"""
import numpy as np
import pandas as pd


def run_backtest(weights, close, cost_bps=0.0, ann=242):
    """
    weights: 目标权重宽表（T 日收盘生成），行=日期，列=股票。
    close: 收盘价宽表。
    假设 T 日收盘调仓、T+1 全天持有（用 shift 避免未来函数）。
    返回 dict: return, nav, turnover, cost, gross_return。
    """
    weights = weights.reindex(index=close.index, columns=close.columns).fillna(0)
    ret = close.pct_change().fillna(0)
    # 昨日收盘权重 × 今日收益
    gross = (weights.shift(1) * ret).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    cost = turnover * (cost_bps / 10000.0)
    net = gross - cost
    return {
        "return": net,
        "gross_return": gross,
        "nav": (1 + net).cumprod(),
        "turnover": turnover,
        "cost": cost,
    }


def performance_summary(r, bench=None, ann=242):
    """常用绩效指标。"""
    n = len(r)
    ann_ret = (1 + r).prod() ** (ann / n) - 1
    ann_vol = r.std() * np.sqrt(ann)
    nav = (1 + r).cumprod()
    mdd = (nav / nav.cummax() - 1).min()
    out = {
        "年化收益": ann_ret,
        "年化波动": ann_vol,
        "夏普": ann_ret / ann_vol if ann_vol > 0 else np.nan,
        "最大回撤": mdd,
        "Calmar": ann_ret / abs(mdd) if mdd < 0 else np.nan,
        "胜率": (r > 0).mean(),
    }
    if bench is not None:
        active = (r - bench).dropna()
        out["信息比率"] = (active.mean() / active.std() * np.sqrt(ann)
                          if active.std() > 0 else np.nan)
    return pd.Series(out)
