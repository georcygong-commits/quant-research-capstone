# -*- coding: utf-8 -*-
"""
风险平价与层次风险平价（M4.4）：
等风险贡献（ERC）风险平价、Lopez de Prado(2016) 层次风险平价（HRP），
并提供风险贡献分解，便于与等权/最小方差对比。
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def risk_contributions(w, Sigma):
    """
    各资产对组合方差的风险贡献（归一化后和为 1）：
        RC_i = w_i (Sigma w)_i / (w' Sigma w)
    """
    w = np.asarray(w, float)
    S = np.asarray(Sigma, float)
    rc = w * (S @ w)
    return rc / rc.sum()


def erc_weights(Sigma):
    """
    等风险贡献（风险平价）：最小化各资产风险贡献的离散度，
    约束权重和为 1、非负。
    """
    S = np.asarray(Sigma, float)
    n = S.shape[0]

    def obj(w):
        rc = w * (S @ w)
        return np.sum((rc - rc.mean()) ** 2)

    res = minimize(obj, np.ones(n) / n, method="SLSQP",
                   bounds=[(0.0, 1.0)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}])
    idx = Sigma.index if isinstance(Sigma, pd.DataFrame) else None
    return pd.Series(res.x, index=idx)


def _correl_dist(corr):
    """相关性 -> 距离矩阵：d = sqrt(0.5(1-corr))，范围 [0,1]。"""
    d = np.sqrt(np.clip(0.5 * (1.0 - np.asarray(corr, float)), 0, 1))
    np.fill_diagonal(d, 0.0)
    return d


def _get_quasi_diag(link):
    """准对角化：将 linkage 展开为资产的线性顺序。"""
    link = link.astype(int)
    n = link.shape[0] + 1
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    while sort_ix.max() >= n:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= n]
        i = df0.index
        j = df0.values - n
        sort_ix[i] = link[j, 0]
        df1 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df1]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()


def _cluster_var(cov, items):
    """簇内逆方差权重组合的方差。"""
    sub = np.asarray(cov.loc[items, items], float)
    ivp = 1.0 / np.diag(sub)
    ivp = ivp / ivp.sum()
    return float(ivp @ sub @ ivp)


def _recursive_bisection(cov, order):
    """递归二分：按两簇逆方差组合方差分配权重。"""
    w = pd.Series(1.0, index=order)
    clusters = [order]
    while clusters:
        clusters = [c[j:k] for c in clusters
                    for j, k in ((0, len(c) // 2), (len(c) // 2, len(c)))
                    if len(c) > 1]
        for i in range(0, len(clusters), 2):
            c0, c1 = clusters[i], clusters[i + 1]
            v0 = _cluster_var(cov, c0)
            v1 = _cluster_var(cov, c1)
            alpha = 1.0 - v0 / (v0 + v1)
            w[c0] *= alpha
            w[c1] *= 1.0 - alpha
    return w


def hrp_weights(returns):
    """
    层次风险平价（HRP）三步：
    1) 相关性转距离并层次聚类；2) 准对角化排序；3) 递归二分配置。
    不要求协方差可逆，在高维/病态下仍稳健。
    """
    R = pd.DataFrame(returns).dropna(axis=1, how="all")
    cov = R.cov()
    corr = R.corr()
    dist = _correl_dist(corr)
    link = linkage(squareform(dist, checks=False), method="single")
    order = _get_quasi_diag(link)
    order = [R.columns[i] for i in order]
    w = _recursive_bisection(cov, order)
    return w.reindex(R.columns).fillna(0.0), link, order
