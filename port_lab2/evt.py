# -*- coding: utf-8 -*-
"""
尾部风险与压力测试（M4.8）：
历史法/正态法 VaR 与 ES，超阈值（POT）广义帕累托 GPD 尾部建模，
以及情景压力测试。
"""
import numpy as np
import pandas as pd
from scipy import stats


def _loss(returns):
    """转为损失序列（正数代表亏损），去缺失。"""
    r = np.asarray(returns, float)
    r = r[~np.isnan(r)]
    return -r


def historical_var_es(returns, alpha=0.05):
    """历史模拟法：VaR 为损失的 (1-alpha) 分位，ES 为超过 VaR 的均值。"""
    L = _loss(returns)
    var = np.quantile(L, 1 - alpha)
    es = L[L >= var].mean()
    return {"VaR": float(var), "ES": float(es)}


def normal_var_es(returns, alpha=0.05):
    """正态假设：用样本均值/标准差解析 VaR、ES。"""
    r = np.asarray(returns, float)
    r = r[~np.isnan(r)]
    mu, sd = r.mean(), r.std(ddof=1)
    z = stats.norm.ppf(1 - alpha)
    var = -(mu - z * sd)
    es = -(mu - sd * stats.norm.pdf(z) / alpha)
    return {"VaR": float(var), "ES": float(es)}


def gpd_fit(returns, quantile=0.95):
    """
    POT：对损失超阈值部分拟合广义帕累托分布（loc 固定 0）。
    返回形状 xi、尺度 beta、阈值 u 及超阈值个数。
    """
    L = _loss(returns)
    u = np.quantile(L, quantile)
    excess = L[L > u] - u
    xi, _, beta = stats.genpareto.fit(excess, floc=0)
    return {"xi": float(xi), "beta": float(beta), "u": float(u),
            "n_total": len(L), "n_excess": len(excess)}


def gpd_var_es(returns, alpha=0.01, quantile=0.95):
    """
    GPD(POT) 尾部 VaR 与 ES：
      VaR = u + beta/xi [ (n/Nu * alpha)^(-xi) - 1 ]
      ES  = VaR/(1-xi) + (beta - xi u)/(1-xi)
    alpha 为尾部概率（0.01 即 99% VaR）。
    """
    g = gpd_fit(returns, quantile)
    xi, beta, u = g["xi"], g["beta"], g["u"]
    n, nu = g["n_total"], g["n_excess"]
    if abs(xi) < 1e-4:                    # 指数分布特例
        var = u + beta * np.log(nu / (n * alpha))
        es = var + beta
    else:
        var = u + (beta / xi) * ((n / nu * alpha) ** (-xi) - 1.0)
        es = var / (1 - xi) + (beta - xi * u) / (1 - xi)
    g.update({"VaR": float(var), "ES": float(es)})
    return g


def stress_test(weights, shocks, daily_vol=None):
    """
    情景压力测试：给定各资产的情景冲击（百分比，负数为下跌），
    计算组合层面的损益。shocks 为 dict/Series 或单一共同冲击。
    """
    w = np.asarray(weights, float)
    if np.isscalar(shocks):
        s = np.full(len(w), shocks)
    else:
        s = np.asarray(pd.Series(shocks).reindex(weights.index).fillna(0.0))
    pnl = float(w @ s)
    return {"portfolio_pnl": pnl, "worst_asset": float(np.min(s)),
            "n_assets_hit": int((s < 0).sum())}
