# -*- coding: utf-8 -*-
"""
统计显著性与 bootstrap（M5.8）：
收益均值 t 检验、Alpha/Beta 的 numpy 与 statsmodels 双实现对照、
iid 与 block bootstrap（保留自相关）置信区间。
"""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def mean_t_test(returns, h0=0.0):
    """单样本均值 t 检验（收益是否显著异于 0）。"""
    r = np.asarray(returns, float)
    n = len(r)
    t_stat = (r.mean() - h0) / (r.std(ddof=1) / np.sqrt(n))
    p = 2 * stats.t.sf(abs(t_stat), df=n - 1)
    return {"mean": float(r.mean()), "t": float(t_stat),
            "p": float(p), "n": n}


def alpha_beta_numpy(returns, bench):
    """CAPM 回归 numpy 实现：r = alpha + beta b + eps，返回 alpha/beta 及其 t 值。"""
    df = pd.concat([pd.Series(returns), pd.Series(bench)], axis=1).dropna()
    df.columns = ["r", "b"]
    X = np.column_stack([np.ones(len(df)), df["b"].values])
    y = df["r"].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - 2
    sigma2 = (resid ** 2).sum() / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    return {"alpha": float(beta[0]), "beta": float(beta[1]),
            "alpha_t": float(beta[0] / se[0]), "beta_t": float(beta[1] / se[1])}


def alpha_beta_statsmodels(returns, bench):
    """同一 CAPM 回归的 statsmodels 实现，用于与 numpy 结果对照。"""
    df = pd.concat([pd.Series(returns), pd.Series(bench)], axis=1).dropna()
    df.columns = ["r", "b"]
    X = sm.add_constant(df["b"].values)
    model = sm.OLS(df["r"].values, X).fit()
    return {"alpha": float(model.params[0]), "beta": float(model.params[1]),
            "alpha_t": float(model.tvalues[0]), "beta_t": float(model.tvalues[1])}


def iid_bootstrap(returns, n_boot=1000, seed=42):
    """iid bootstrap：年化 Sharpe 分布与 95% 区间、Sharpe>0 概率。"""
    rng = np.random.default_rng(seed)
    r = np.asarray(returns, float); n = len(r)
    srs = []
    for _ in range(n_boot):
        x = r[rng.integers(0, n, n)]
        srs.append(x.mean() / x.std() * np.sqrt(242))
    srs = np.array(srs)
    return {"ci_low": float(np.quantile(srs, 0.025)),
            "ci_high": float(np.quantile(srs, 0.975)),
            "prob_positive": float(np.mean(srs > 0))}


def block_bootstrap(returns, block_len=20, n_boot=1000, seed=42):
    """移动块 bootstrap：保留收益的时间自相关，给 Sharpe 区间。"""
    rng = np.random.default_rng(seed)
    r = np.asarray(returns, float); n = len(r)
    n_blocks = int(np.ceil(n / block_len))
    srs = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block_len + 1, n_blocks)
        sample = np.concatenate([r[s:s + block_len] for s in starts])[:n]
        srs.append(sample.mean() / sample.std() * np.sqrt(242))
    srs = np.array(srs)
    return {"ci_low": float(np.quantile(srs, 0.025)),
            "ci_high": float(np.quantile(srs, 0.975)),
            "prob_positive": float(np.mean(srs > 0)),
            "block_len": block_len}
