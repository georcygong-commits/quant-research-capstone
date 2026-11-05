# -*- coding: utf-8 -*-
"""均值-方差优化（cvxpy）：有效前沿、最小方差、最大夏普（从前沿点选）。"""
import numpy as np
import pandas as pd
import cvxpy as cp


def min_variance(mu, Sigma, target_ret, long_only=True):
    """在期望收益 >= target_ret 下最小化方差，返回权重。"""
    n = len(mu)
    w = cp.Variable(n)
    cons = [cp.sum(w) == 1, mu @ w >= target_ret]
    if long_only:
        cons.append(w >= 0)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(Sigma))), cons)
    prob.solve(solver=cp.CLARABEL)
    return w.value


def efficient_frontier(mu, Sigma, n_points=15, lo=None, hi=None,
                       long_only=True):
    """返回有效前沿点列表：(期望收益, 波动, 权重)。"""
    lo = np.quantile(mu, 0.3) if lo is None else lo
    hi = np.quantile(mu, 0.95) if hi is None else hi
    pts = []
    for t in np.linspace(lo, hi, n_points):
        w = min_variance(mu, Sigma, t, long_only)
        if w is not None:
            pts.append((mu @ w, np.sqrt(w @ Sigma @ w), w))
    return pts


def max_sharpe(frontier):
    """从有效前沿点中选夏普最高的，返回 (夏普, 收益, 波动, 权重)。"""
    best = max(frontier, key=lambda p: p[0] / p[1] if p[1] > 0 else -1)
    r, v, w = best
    return r / v, r, v, w


def min_variance_portfolio(Sigma, long_only=True):
    """全局最小方差组合。"""
    n = Sigma.shape[0]
    w = cp.Variable(n)
    cons = [cp.sum(w) == 1]
    if long_only:
        cons.append(w >= 0)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(Sigma))), cons)
    prob.solve(solver=cp.CLARABEL)
    return w.value
