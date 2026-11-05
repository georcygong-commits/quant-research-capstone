# -*- coding: utf-8 -*-
"""
纯因子组合：通过截面加权回归构造只暴露于单一因子的组合。
"""
import numpy as np
import pandas as pd


def pure_factor_weights(X, weights=None):
    """
    X: 单截面暴露矩阵（股票 × 因子，含截距列），weights: 股票权重（市值等）。
    返回纯因子组合权重（股票 × 因子），每列对应该因子暴露1、其余因子暴露0。
    """
    Xv = np.asarray(X, float)
    if weights is None:
        W = np.eye(len(Xv))
    else:
        W = np.diag(np.asarray(weights, float))
    XtWX = Xv.T @ W @ Xv
    inv = np.linalg.pinv(XtWX)
    P = W @ Xv @ inv            # 纯因子组合权重
    return pd.DataFrame(P, index=X.index, columns=X.columns)


def factor_returns(X, y, weights=None):
    """单截面加权最小二乘因子收益。"""
    P = pure_factor_weights(X, weights)
    return P.T @ y


def pure_portfolio_exposure_check(X, weights=None):
    """验证纯因子组合的暴露矩阵应为单位阵。"""
    P = pure_factor_weights(X, weights)
    return np.asarray(X, float).T @ P.values
