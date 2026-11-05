# -*- coding: utf-8 -*-
"""
Barra 风格归因（M5.7）：
组合在风格因子上的暴露、截面回归因子收益、因子贡献与特定收益，
时间序列累计并核对归因闭合。
"""
import numpy as np
import pandas as pd


def portfolio_exposures(weights, factor_panels):
    """组合因子暴露 = 个股权重 × 个股暴露的截面和。"""
    out = {}
    for name, panel in factor_panels.items():
        p = panel.reindex(index=weights.index, columns=weights.columns)
        out[name] = (weights * p).sum(axis=1)
    return pd.DataFrame(out)


def _cs_regression(y, X):
    """单截面 OLS：返回因子收益与残差。"""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return beta, resid


def style_attribution(weights, returns, factor_panels):
    """
    weights: 组合权重宽表；returns: 个股收益宽表；
    factor_panels: {因子名: 暴露宽表}。
    逐期截面回归（含国家/截距因子），计算组合暴露、因子贡献与特定贡献，
    返回累计贡献、闭合残差。
    """
    names = list(factor_panels.keys())
    dates = returns.index
    port_exp = portfolio_exposures(weights, factor_panels)
    contrib = pd.DataFrame(index=dates, columns=["Country"] + names + ["Specific"],
                           data=0.0)
    for t in dates:
        cols = returns.columns
        y = np.asarray(returns.loc[t], float)
        X_cols = [np.ones(len(cols))]
        for k in names:
            vals = np.asarray(factor_panels[k].reindex(columns=cols).loc[t], float)
            X_cols.append(np.nan_to_num(vals, nan=0.0))
        X = np.vstack(X_cols).T
        mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        if mask.sum() <= len(names) + 1:
            continue
        beta, resid = _cs_regression(y[mask], X[mask])
        w = np.asarray(weights.loc[t], float)[mask]
        gross = w.sum()
        contrib.loc[t, "Country"] = beta[0] * gross
        for j, k in enumerate(names):
            contrib.loc[t, k] = port_exp.loc[t, k] * beta[j + 1]
        contrib.loc[t, "Specific"] = float(np.sum(w * resid))
    actual = (weights * returns).sum(axis=1)
    cum = (1 + contrib).cumprod() - 1
    return {"contrib": contrib, "cum_contrib": cum,
            "port_exposures": port_exp,
            "actual_period_sum": float(actual.sum()),
            "attrib_period_sum": float(contrib.sum(axis=1).sum()),
            "closure_residual": float(actual.sum() - contrib.sum(axis=1).sum())}
