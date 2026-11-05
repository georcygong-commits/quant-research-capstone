# -*- coding: utf-8 -*-
"""
Black-Litterman 模型（M4.3）：
逆优化求市场均衡收益，结合主观观点（P, Q, Omega）得到后验收益，
再做均值-方差优化，得到融合市场共识与主观判断的配置权重。
"""
import numpy as np
import pandas as pd


def equilibrium_returns(Sigma, w_mkt, delta=2.5):
    """
    逆优化：市场组合隐含的均衡收益 pi = delta Sigma w_mkt。
    delta 为风险厌恶系数（通常取 2.5 左右）。
    """
    S = np.asarray(Sigma, float)
    w = np.asarray(w_mkt, float)
    pi = delta * S @ w
    if isinstance(Sigma, pd.DataFrame):
        return pd.Series(pi, index=Sigma.index)
    return pi


def default_omega(Sigma, P, tau):
    """
    观点不确定矩阵的默认取法：Omega = diag(tau P Sigma P')，
    使各观点信心与该观点组合的方差成比例。
    """
    S = np.asarray(Sigma, float)
    P = np.atleast_2d(np.asarray(P, float))
    diag = np.diag(tau * P @ S @ P.T)
    return np.diag(diag)


def bl_posterior(Sigma, pi, P, Q, tau=0.05, omega=None, delta=2.5):
    """
    Black-Litterman 后验收益与后验协方差：
      E[R] = [(tau S)^-1 + P'Omega^-1 P]^-1
             [(tau S)^-1 pi + P'Omega^-1 Q]
      M     = [(tau S)^-1 + P'Omega^-1 P]^-1
      Sigma_post = S + M
    返回 dict：后验收益 Series、后验协方差、使用的 Omega。
    """
    S = np.asarray(Sigma, float)
    pi = np.asarray(pi, float)
    P = np.atleast_2d(np.asarray(P, float))
    Q = np.asarray(Q, float)
    if omega is None:
        omega = default_omega(S, P, tau)
    omega = np.atleast_2d(np.asarray(omega, float))

    tauS = tau * S
    inv_tauS = np.linalg.pinv(tauS)
    inv_om = np.linalg.pinv(omega)
    M = np.linalg.pinv(inv_tauS + P.T @ inv_om @ P)
    er = M @ (inv_tauS @ pi + P.T @ inv_om @ Q)
    Sigma_post = S + M

    idx = Sigma.index if isinstance(Sigma, pd.DataFrame) else None
    return {"er": pd.Series(er, index=idx),
            "Sigma_post": pd.DataFrame(Sigma_post, index=idx, columns=idx),
            "omega": pd.DataFrame(omega), "M": pd.DataFrame(M, index=idx, columns=idx)}


def bl_weights(Sigma, expected_returns, delta=2.5, long_only=False):
    """
    由后验收益求最优权重：
      无约束解析解 w* = (delta Sigma)^-1 E[R]（再归一化）；
      long_only=True 时用 cvxpy 求非负最优解。
    """
    S = np.asarray(Sigma, float)
    er = np.asarray(expected_returns, float)
    if long_only:
        import cvxpy as cp
        w = cp.Variable(len(er))
        prob = cp.Problem(cp.Maximize(er @ w - 0.5 * delta *
                                      cp.quad_form(w, cp.psd_wrap(S))),
                          [cp.sum(w) == 1, w >= 0])
        prob.solve(solver=cp.CLARABEL)
        wv = w.value
    else:
        wv = np.linalg.solve(delta * S, er)
        wv = wv / wv.sum()
    idx = Sigma.index if isinstance(Sigma, pd.DataFrame) else None
    return pd.Series(wv, index=idx)
