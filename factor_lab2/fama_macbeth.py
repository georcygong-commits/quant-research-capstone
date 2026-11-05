# -*- coding: utf-8 -*-
"""
Fama-MacBeth 两步回归（截面风险溢价），statsmodels 与 numpy 双实现。
给定暴露面板，逐截面回归估计因子溢价，再对溢价序列做 t 检验。
"""
import numpy as np
import pandas as pd


def _t_stats(gamma_df):
    T = len(gamma_df)
    return gamma_df.mean() / (gamma_df.std() / np.sqrt(T))


def fama_macbeth_numpy(returns, exposures):
    """numpy 手写：逐截面 lstsq，返回 (溢价均值, t值, 溢价序列)。"""
    names = list(exposures.keys())
    rows = {}
    for dt in returns.index:
        y = returns.loc[dt]
        X = pd.DataFrame({k: exposures[k].loc[dt] for k in names})
        ok = y.notna() & X.notna().all(axis=1)
        if ok.sum() < len(names) + 5:
            continue
        Xv = np.column_stack([np.ones(ok.sum()), X[ok].values])
        gamma, *_ = np.linalg.lstsq(Xv, y[ok].values, rcond=None)
        rows[dt] = gamma
    g = pd.DataFrame(rows, index=["const"] + names).T
    return g.mean(), _t_stats(g), g


def fama_macbeth_statsmodels(returns, exposures):
    """statsmodels 实现：逐截面 OLS，返回 (溢价均值, t值, 溢价序列)。"""
    import statsmodels.api as sm
    names = list(exposures.keys())
    rows = {}
    for dt in returns.index:
        y = returns.loc[dt]
        X = pd.DataFrame({k: exposures[k].loc[dt] for k in names})
        ok = y.notna() & X.notna().all(axis=1)
        if ok.sum() < len(names) + 5:
            continue
        Xv = sm.add_constant(X[ok].values)
        model = sm.OLS(y[ok].values, Xv).fit()
        rows[dt] = model.params
    g = pd.DataFrame(rows, index=["const"] + names).T
    return g.mean(), _t_stats(g), g
