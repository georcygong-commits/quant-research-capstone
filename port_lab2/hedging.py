# -*- coding: utf-8 -*-
"""
动态风险管理与对冲（M4.9）：
波动率目标（vol targeting）、最小方差对冲比率、对冲后风险、
回撤止损/去杠杆触发与动态杠杆。
"""
import numpy as np
import pandas as pd


def vol_target_leverage(realized_ann_vol, target_ann_vol=0.12, max_lev=1.5):
    """目标杠杆 = 目标波动/已实现波动，并封顶，防止低波期过度加杠杆。"""
    if realized_ann_vol <= 0:
        return max_lev
    return float(min(max_lev, target_ann_vol / realized_ann_vol))


def apply_vol_target(returns, target_vol=0.12, window=20, max_lev=1.5):
    """
    波动率目标路径：滚动已实现年化波动 -> 目标杠杆（shift(1) 防未来函数）
    -> 目标策略收益。高波期自动降杠杆，低波期加杠杆。
    """
    r = pd.Series(returns).dropna()
    realized = r.rolling(window).std() * np.sqrt(242)
    lev = (target_vol / realized).clip(0, max_lev).shift(1)
    strat = (r * lev).dropna()
    return strat, lev


def min_variance_hedge(asset_ret, hedge_ret):
    """
    最小方差对冲比率 beta* = Cov(r_a, r_h)/Var(r_h)，
    并报告对冲前后波动率与残差。
    """
    df = pd.DataFrame({"a": asset_ret, "h": hedge_ret}).dropna()
    cov = np.cov(df["a"], df["h"], ddof=1)
    beta = cov[0, 1] / cov[1, 1]
    resid = df["a"] - beta * df["h"]
    return {"beta": float(beta),
            "raw_vol": float(df["a"].std() * np.sqrt(242)),
            "hedged_vol": float(resid.std() * np.sqrt(242)),
            "vol_reduction": float(1 - resid.std() / df["a"].std())}


def drawdown_stop(equity, dd_limit=0.15):
    """
    回撤止损/去杠杆：净值相对历史峰值回撤首次触及 dd_limit 即触发。
    返回首次触发位置、最大回撤与触发次数。
    """
    eq = pd.Series(equity)
    peak = eq.cummax()
    dd = eq / peak - 1.0
    breach = dd <= -dd_limit
    first = breach.idxmax() if breach.any() else None
    return {"first_trigger": first, "max_drawdown": float(dd.min()),
            "n_breach": int(breach.sum())}


def leverage_from_margin(equity, debt):
    """总杠杆倍数 = (权益+负债)/权益。"""
    return float((equity + debt) / equity)
