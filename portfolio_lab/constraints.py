# -*- coding: utf-8 -*-
"""带约束的组合优化：单票/行业/风格/换手/风险约束（cvxpy）。"""
import numpy as np
import pandas as pd
import cvxpy as cp


def optimize(alpha, Sigma, benchmark=None, max_weight=0.05,
             industry=None, industry_dev=None,
             factor_exposures=None, factor_dev=None,
             max_turnover=None, w_prev=None,
             risk_aversion=1.0, gross=1.0, long_only=True):
    """
    最大化均值-方差效用：alpha^T w - (ra/2) w^T Sigma w。
    - alpha: 预期收益（Series）；Sigma: 协方差（DataFrame）
    - benchmark: 基准权重（用于行业/风格中性基准）
    - industry: 行业标签（Series）；industry_dev: 行业偏离上限
    - factor_exposures: {因子: 暴露Series}；factor_dev: 风格偏离上限
    - max_turnover: 单边换手率上界（L1）；w_prev: 上一期权重
    """
    idx = alpha.index
    a = alpha.values
    S = cp.psd_wrap(Sigma.reindex(index=idx, columns=idx).values)
    n = len(idx)
    w = cp.Variable(n)
    cons = [cp.sum(w) == gross]
    if long_only:
        cons.append(w >= 0)
    if max_weight is not None:
        cons.append(w <= max_weight)
    if benchmark is None:
        benchmark = pd.Series(1.0 / n, index=idx)

    if industry is not None and industry_dev is not None:
        ind = industry.reindex(idx)
        for v in ind.dropna().unique():
            m = (ind == v).values.astype(float)
            b = m @ benchmark.reindex(idx).values
            cons.append(m @ w <= b + industry_dev)
            cons.append(m @ w >= b - industry_dev)

    if factor_exposures is not None and factor_dev is not None:
        for k, exp in factor_exposures.items():
            e = exp.reindex(idx).fillna(0).values
            b = e @ benchmark.reindex(idx).values
            cons.append(e @ w <= b + factor_dev)
            cons.append(e @ w >= b - factor_dev)

    if max_turnover is not None and w_prev is not None:
        cons.append(cp.norm(w - w_prev.reindex(idx).values, 1) <= max_turnover)

    obj = cp.Maximize(a @ w - risk_aversion / 2 * cp.quad_form(w, S))
    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.CLARABEL)
    if w.value is None:
        return None
    return pd.Series(w.value, index=idx)


def turnover(w_new, w_prev):
    """单边换手率（0.5 * L1）。"""
    common = w_new.index.union(w_prev.index)
    d = (w_new.reindex(common).fillna(0) - w_prev.reindex(common).fillna(0)).abs()
    return 0.5 * d.sum()
