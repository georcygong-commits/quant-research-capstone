# -*- coding: utf-8 -*-
"""
鲁棒优化与估计误差（M4.5）：
期望收益 James-Stein 收缩、Michaud(1998) 重采样投资组合、
协方差不确定下的对角加载（worst-case）稳健最小方差，以及前沿不确定性。
"""
import numpy as np
import pandas as pd
import cvxpy as cp


def _arr(returns):
    if isinstance(returns, pd.DataFrame):
        X = returns.dropna(axis=1, how="all")
        return X, X.columns.tolist()
    return pd.DataFrame(returns), list(range(np.asarray(returns).shape[1]))


def shrink_mu(returns):
    """
    James-Stein 式期望收益收缩：把各资产历史均值向全局 grand mean 拉回，
    降低期望收益估计的高方差。返回收缩后的年化期望收益。
    """
    R, cols = _arr(returns)
    mu = R.mean(axis=0).values
    grand = mu.mean()
    T = len(R)
    m = len(cols)
    s2 = np.mean(R.var(axis=0, ddof=1).values)     # 平均抽样方差
    # 标准 James-Stein 收缩因子 B（截断到 [0,1]）
    ss = np.sum((mu - grand) ** 2)
    B = 0.0 if ss <= 0 else (m - 2.0) * s2 / (T * ss)
    shrink = float(np.clip(B, 0.0, 1.0))
    mu_js = grand + (1.0 - shrink) * (mu - grand)
    return pd.Series(mu_js * 242, index=cols), shrink, float(B)


def _minvar_target(Sigma, target, mu):
    """给定目标收益的最小方差权重（long-only）。"""
    n = len(mu)
    w = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(Sigma))),
                      [cp.sum(w) == 1, w >= 0, mu @ w >= target])
    prob.solve(solver=cp.CLARABEL)
    return w.value


def resampled_weights(returns, n_sims=60, seed=42):
    """
    Michaud 重采样：bootstrap 重抽历史收益，每次重新估计 mu/Sigma 并
    求同一目标收益下的最小方差权重，最后对所有权重平均。
    平均后权重更分散，降低对单一历史路径的依赖。
    """
    rng = np.random.default_rng(seed)
    R, cols = _arr(returns)
    X = R.values
    T, n = X.shape
    target = np.quantile(X.mean(axis=0), 0.7)
    W = np.zeros((n_sims, n))
    ok = 0
    for b in range(n_sims):
        idx = rng.integers(0, T, T)
        Xb = X[idx]
        mu = Xb.mean(axis=0)
        S = np.cov(Xb, rowvar=False)
        wv = _minvar_target(S, target, mu)
        if wv is not None:
            W[ok] = wv
            ok += 1
    W = W[:ok]
    return pd.Series(W.mean(axis=0), index=cols), W


def robust_min_variance(returns, epsilon=None):
    """
    协方差不确定下的稳健最小方差：对角加载 Sigma + epsilon I，
    等价于对协方差估计误差做 worst-case 保护，抑制小特征值噪声。
    epsilon 默认取样本方差中位数的 5%。
    """
    R, cols = _arr(returns)
    X = R.values
    S = np.cov(X, rowvar=False)
    if epsilon is None:
        epsilon = 0.05 * np.median(np.diag(S))
    S_rob = S + epsilon * np.eye(len(cols))
    w = cp.Variable(len(cols))
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(S_rob))),
                      [cp.sum(w) == 1, w >= 0])
    prob.solve(solver=cp.CLARABEL)
    return pd.Series(w.value, index=cols), float(epsilon), np.linalg.cond(S_rob)


def frontier_uncertainty(returns, n_sims=60, n_targets=10, seed=42):
    """
    bootstrap 估计有效前沿的均值与 5%/95% 分位带，
    展示样本内最优收益-风险的估计不确定性。
    """
    rng = np.random.default_rng(seed)
    R, _ = _arr(returns)
    X = R.values
    T, n = X.shape
    mu0 = X.mean(axis=0)
    targets = np.linspace(np.quantile(mu0, 0.35), np.quantile(mu0, 0.9), n_targets)
    vols = np.zeros((n_sims, n_targets))
    for b in range(n_sims):
        idx = rng.integers(0, T, T)
        Xb = X[idx]
        Sb = np.cov(Xb, rowvar=False)
        mub = Xb.mean(axis=0)
        for j, t in enumerate(targets):
            wv = _minvar_target(Sb, t, mub)
            vols[b, j] = np.sqrt(wv @ Sb @ wv) if wv is not None else np.nan
    return {"targets": targets, "vol_mean": np.nanmean(vols, axis=0),
            "vol_p05": np.nanquantile(vols, 0.05, axis=0),
            "vol_p95": np.nanquantile(vols, 0.95, axis=0)}
