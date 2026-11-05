# -*- coding: utf-8 -*-
"""
机构级协方差估计（M4.1）：
样本协方差的病态问题、Ledoit-Wolf 收缩、随机矩阵理论（RMT）去噪、
因子模型协方差，以及条件数 / Frobenius 稳定性对比。
"""
import numpy as np
import pandas as pd


def _to_array(returns):
    """收益宽表 -> 中心化 ndarray（逐列去均值），并保留列名。"""
    if isinstance(returns, pd.DataFrame):
        cols = returns.columns.tolist()
        X = returns.dropna(axis=1, how="all").values
    else:
        cols = list(range(np.asarray(returns).shape[1]))
        X = np.asarray(returns, float)
    X = X[~np.isnan(X).any(axis=1)]          # 删除含缺失的行
    X = X - X.mean(axis=0, keepdims=True)    # 中心化
    return X, cols


def sample_cov(returns, ann=None):
    """无偏样本协方差（分母 T-1）。ann 给定时按年因子缩放。"""
    X, cols = _to_array(returns)
    S = np.cov(X, rowvar=False, ddof=1)
    if ann is not None:
        S = S * ann
    return pd.DataFrame(S, index=cols, columns=cols)


def condition_number(returns):
    """样本协方差的条件数（最大/最小特征值），越大越病态。"""
    X, _ = _to_array(returns)
    S = np.cov(X, rowvar=False, ddof=1)
    w = np.linalg.eigvalsh(S)
    w = w[w > 1e-18]
    return float(w.max() / w.min())


def ledoit_wolf_shrink(returns):
    """
    自实现 Ledoit-Wolf(2004) 收缩：收缩到「平均方差 × 单位阵」。
        Sigma_shrink = (1-delta) S + delta F,  F = tr(S)/n I
    收缩强度 delta 由最小化 Frobenius 风险的闭式解给出。
    返回 (协方差 DataFrame, 收缩强度 delta)。
    """
    X, cols = _to_array(returns)
    T, n = X.shape
    S = (X.T @ X) / T                       # MLE 样本协方差（分母 T）
    F = np.trace(S) / n * np.eye(n)         # 结构化目标：各向同性
    d = np.sum((S - F) ** 2)                # 目标与样本的偏离
    # 向量化计算 b_bar = (1/T^2) sum_t ||x_t x_t' - S||_F^2
    q = np.sum(X * X, axis=1)               # 每行 ||x_t||^2
    term1 = np.sum(q * q)
    term2 = np.sum(X * (X @ S))             # sum_t x_t' S x_t
    term3 = T * np.trace(S @ S)
    b_bar = (term1 - 2.0 * term2 + term3) / (T * T)
    b = min(b_bar, d)
    delta = 0.0 if d <= 0 else b / d
    Sigma = (1.0 - delta) * S + delta * F
    return pd.DataFrame(Sigma, index=cols, columns=cols), float(delta)


def ledoit_wolf_sklearn(returns):
    """sklearn Ledoit-Wolf 对照（收缩目标为对角样本方差）。"""
    from sklearn.covariance import LedoitWolf
    X, cols = _to_array(returns)
    lw = LedoitWolf(assume_centered=True).fit(X)
    return (pd.DataFrame(lw.covariance_, index=cols, columns=cols),
            float(lw.shrinkage_))


def marchenko_pastur_bounds(returns):
    """
    Marchenko-Pastur 随机矩阵理论噪声特征值边界：
        q = n/T,  lambda_± = (1 ± sqrt(q))^2 （相关矩阵口径，方差取 1）
    """
    X, _ = _to_array(returns)
    T, n = X.shape
    q = n / T
    lam_minus = (1.0 - np.sqrt(q)) ** 2
    lam_plus = (1.0 + np.sqrt(q)) ** 2
    return float(lam_minus), float(lam_plus), float(q)


def rmt_denoise(returns, method="constant"):
    """
    RMT 去噪（Laloux 2000 / Bouchaud 常数残差特征值法）：
    1) 对相关矩阵做特征分解；
    2) 低于 MP 上界 lambda+ 的特征值视为噪声，整体替换为其均值；
    3) 重建并重新缩放为相关矩阵（对角线归 1）。
    method='constant' 常数替换；'zero' 直接保留（仅诊断用）。
    返回 dict：去噪相关矩阵、原始/清洗特征值、信号/噪声个数、lambda+。
    """
    X, cols = _to_array(returns)
    T, n = X.shape
    R = np.corrcoef(X, rowvar=False)
    _, lam_plus, q = marchenko_pastur_bounds(returns)
    w, v = np.linalg.eigh(R)
    noise = w < lam_plus
    w_clean = w.copy()
    if method == "constant" and noise.sum() > 0:
        w_clean[noise] = w[noise].mean()
    R_d = v @ np.diag(w_clean) @ v.T
    d = np.sqrt(np.diag(R_d))              # 重新缩放为相关矩阵
    R_d = R_d / np.outer(d, d)
    return {"corr": pd.DataFrame(R_d, index=cols, columns=cols),
            "eig_raw": w, "eig_clean": w_clean,
            "n_signal": int((~noise).sum()), "n_noise": int(noise.sum()),
            "lam_plus": lam_plus, "q": q}


def factor_model_cov(returns, factor_returns):
    """
    单/多因子模型协方差：r = B f + e  =>  Sigma = B Sigma_f B' + D。
    factor_returns: 因子收益宽表（T×k，需与 returns 日期对齐）。
    """
    if isinstance(returns, pd.DataFrame):
        R = returns.loc[factor_returns.index].dropna(axis=1, how="all")
        F = factor_returns.loc[R.index].fillna(0.0)
    else:
        R = pd.DataFrame(returns)
        F = pd.DataFrame(factor_returns)
    B = np.linalg.lstsq(
        np.column_stack([np.ones(len(F)), F.values]), R.values, rcond=None)[0]
    resid = R.values - np.column_stack([np.ones(len(F)), F.values]) @ B
    B = B[1:]                               # 去掉截距行
    Sf = np.cov(F.values, rowvar=False, ddof=1)
    Sf = np.atleast_2d(Sf)
    D = np.diag(np.var(resid, axis=0, ddof=1))
    Sigma = B.T @ Sf @ B + D
    return pd.DataFrame(Sigma, index=R.columns, columns=R.columns)


def frob_distance(cov_a, cov_b):
    """两个协方差矩阵的 Frobenius 距离，用于衡量估计的稳定性/差异。"""
    A = np.asarray(cov_a, float)
    B = np.asarray(cov_b, float)
    return float(np.sqrt(np.sum((A - B) ** 2)))
