# -*- coding: utf-8 -*-
"""完整绩效指标：风险调整收益、回撤、交易统计、分年度、Alpha/Beta。"""
import numpy as np
import pandas as pd


def sharpe_ratio(r, rf=0.0, ann=242):
    ex = r - rf / ann
    return ex.mean() / ex.std() * np.sqrt(ann)


def sortino_ratio(r, rf=0.0, ann=242):
    ex = r - rf / ann
    downside = np.minimum(ex, 0)
    dd = np.sqrt((downside ** 2).mean())
    return ex.mean() / dd * np.sqrt(ann)


def calmar_ratio(r, ann=242):
    ann_ret = (1 + r).prod() ** (ann / len(r)) - 1
    nav = (1 + r).cumprod()
    mdd = (nav / nav.cummax() - 1).min()
    return ann_ret / abs(mdd)


def information_ratio(r, b, ann=242):
    active = (r - b).dropna()
    return active.mean() / active.std() * np.sqrt(ann)


def alpha_beta(r, b, rf=0.0, ann=242):
    df = pd.concat([r, b], axis=1).dropna()
    df.columns = ["r", "b"]
    beta = df.r.cov(df.b) / df.b.var()
    alpha = (df.r.mean() - beta * df.b.mean()) * ann
    return alpha, beta


def win_rate(r):
    return (r > 0).mean()


def profit_loss_ratio(r):
    wins = r[r > 0]; losses = r[r < 0]
    if len(losses) == 0 or len(wins) == 0:
        return np.nan
    return wins.mean() / abs(losses.mean())


def expectancy(r):
    """期望值 = 胜率×平均盈利 − 败率×平均亏损。"""
    wins = r[r > 0]; losses = r[r < 0]
    p = len(wins) / len(r)
    aw = wins.mean() if len(wins) else 0
    al = abs(losses.mean()) if len(losses) else 0
    return p * aw - (1 - p) * al


def max_consecutive_losses(r):
    is_loss = (r < 0).astype(int).values
    best = cur = 0
    for x in is_loss:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return best


def yearly_returns(r):
    return (1 + r).groupby(r.index.year).prod() - 1


def monthly_returns(r):
    return (1 + r).groupby([r.index.year, r.index.month]).prod() - 1


def full_report(r, b=None, rf=0.0, ann=242):
    ann_ret = (1 + r).prod() ** (ann / len(r)) - 1
    ann_vol = r.std() * np.sqrt(ann)
    nav = (1 + r).cumprod()
    mdd = (nav / nav.cummax() - 1).min()
    out = {
        "年化收益": ann_ret, "年化波动": ann_vol,
        "夏普": sharpe_ratio(r, rf, ann),
        "Sortino": sortino_ratio(r, rf, ann),
        "Calmar": calmar_ratio(r, ann),
        "最大回撤": mdd,
        "胜率": win_rate(r),
        "盈亏比": profit_loss_ratio(r),
        "期望值": expectancy(r),
        "最大连亏": max_consecutive_losses(r),
    }
    if b is not None:
        a, beta = alpha_beta(r, b, rf, ann)
        out["Alpha"] = a; out["Beta"] = beta
        out["信息比率"] = information_ratio(r, b, ann)
    return pd.Series(out)
