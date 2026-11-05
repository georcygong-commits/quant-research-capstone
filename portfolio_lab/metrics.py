# -*- coding: utf-8 -*-
"""风险指标：波动率、回撤、Beta、VaR/CVaR、集中度、下行风险。"""
import numpy as np
import pandas as pd


def annual_vol(r, ann=242):
    return r.std() * np.sqrt(ann)


def drawdown(r):
    nav = (1 + r).cumprod()
    return nav / nav.cummax() - 1


def max_drawdown(r):
    return drawdown(r).min()


def beta(r, market):
    df = pd.concat([r, market], axis=1).dropna()
    return df.iloc[:, 0].cov(df.iloc[:, 1]) / df.iloc[:, 1].var()


def value_at_risk(r, alpha=0.05):
    """历史 VaR：收益的 alpha 分位数（取负表示损失）。"""
    return -np.quantile(r.dropna(), alpha)


def cvar(r, alpha=0.05):
    """条件 VaR / 期望损失（ES）：尾部损失的平均。"""
    q = np.quantile(r.dropna(), alpha)
    return -r[r <= q].mean()


def hhi(weights):
    """赫芬达尔集中度（越大越集中）。"""
    return (weights ** 2).sum()


def effective_n(weights):
    """有效持仓数（分散化程度）。"""
    return 1.0 / (weights ** 2).sum()


def downside_dev(r, mar=0.0, ann=242):
    """下行标准差（只考虑低于目标的波动）。"""
    d = np.minimum(r - mar, 0)
    return np.sqrt((d ** 2).mean()) * np.sqrt(ann)


def sortino(r, mar=0.0, ann=242):
    ann_ret = (1 + r).prod() ** (ann / len(r)) - 1
    return ann_ret / downside_dev(r, mar, ann)
