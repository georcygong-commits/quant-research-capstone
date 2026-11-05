# -*- coding: utf-8 -*-
"""
动态波动率与协方差（M4.2）：
自实现 GARCH(1,1)（MLE）、arch 库对照、Engle(2002) DCC 动态条件相关。
注：arch 8.0 已移除多元 GARCH，故 DCC 采用 numpy 自实现。
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _clean(returns):
    """单序列去缺失、去均值，返回 ndarray。"""
    r = np.asarray(returns, float)
    r = r[~np.isnan(r)]
    return r - r.mean()


def garch_filter(r, omega, alpha, beta):
    """GARCH(1,1) 方差递推：sigma_t^2 = omega + alpha r_{t-1}^2 + beta sigma_{t-1}^2。"""
    s2 = np.full(len(r), np.var(r))
    for t in range(1, len(r)):
        s2[t] = omega + alpha * r[t - 1] ** 2 + beta * s2[t - 1]
    return s2


def garch11_fit(returns):
    """
    自实现 GARCH(1,1)：高斯准极大似然 + SLSQP 约束优化。
    返回参数、条件方差序列、持续性 alpha+beta。
    """
    r = _clean(returns)
    v0 = np.var(r)

    def nll(p):
        omega, alpha, beta = p
        if alpha + beta >= 1.0:
            return 1e10
        s2 = garch_filter(r, omega, alpha, beta)
        return 0.5 * np.sum(np.log(s2) + r * r / s2)

    res = minimize(nll, [v0 * 0.1, 0.08, 0.90], method="SLSQP",
                   bounds=[(1e-12, None), (0.0, 1.0), (0.0, 1.0)],
                   constraints=[{"type": "ineq",
                                 "fun": lambda x: 1.0 - x[1] - x[2]}])
    omega, alpha, beta = res.x
    s2 = garch_filter(r, omega, alpha, beta)
    return {"omega": float(omega), "alpha": float(alpha), "beta": float(beta),
            "persist": float(alpha + beta), "sigma2": s2, "success": bool(res.success)}


def garch11_forecast(returns, fit, horizon=5):
    """多步条件方差预测（均值回复到无条件方差）。"""
    r = _clean(returns)
    omega, alpha, beta = fit["omega"], fit["alpha"], fit["beta"]
    h = np.zeros(horizon)
    h[0] = omega + alpha * r[-1] ** 2 + beta * fit["sigma2"][-1]
    for i in range(1, horizon):
        h[i] = omega + (alpha + beta) * h[i - 1]
    return h


def garch_archlib(returns):
    """arch 库 GARCH(1,1) 对照（收益放大 100 倍以改善收敛，方差再还原）。"""
    from arch import arch_model
    r = _clean(returns)
    res = arch_model(r * 100.0, mean="Zero", vol="Garch", p=1, q=1,
                     rescale=False).fit(disp="off")
    p = res.params
    return {"omega": float(p["omega"]) / 1e4,
            "alpha": float(p["alpha[1]"]),
            "beta": float(p["beta[1]"]),
            "sigma2": res.conditional_volatility ** 2 / 1e4}


def dcc_fit(returns):
    """
    自实现 Engle(2002) DCC 两步法：
      第一步：逐资产 GARCH(1,1) 得 D_t 与标准化残差 z_t；
      第二步：Q_t=(1-a-b)Qbar + a z_{t-1}z' + b Q_{t-1}，MLE 估计 a,b。
    返回动态相关 Rt、动态协方差 H_t = D_t R_t D_t 及参数。
    建议输入最近 400~600 个交易日、8~12 只资产以控制计算量。
    """
    R = pd.DataFrame(returns).dropna(axis=1, how="all")
    cols = R.columns.tolist()
    r = R.values
    T, n = r.shape
    D = np.zeros((T, n))
    Z = np.zeros((T, n))
    uni = []
    for j in range(n):
        f = garch11_fit(r[:, j])
        D[:, j] = np.sqrt(f["sigma2"])
        Z[:, j] = _clean(r[:, j]) / D[:, j]
        uni.append(f)
    Qbar = np.atleast_2d(np.cov(Z, rowvar=False))

    def dcc_nll(ab):
        a, b = ab
        if a + b >= 1.0:
            return 1e10
        Q = Qbar.copy()
        ll = 0.0
        for t in range(T):
            if t > 0:
                Q = (1 - a - b) * Qbar + a * np.outer(Z[t - 1], Z[t - 1]) + b * Q
            d = np.sqrt(np.diag(Q))
            Rt = Q / np.outer(d, d)
            sign, logdet = np.linalg.slogdet(Rt)
            if sign <= 0:
                return 1e10
            z = Z[t]
            ll += logdet + z @ np.linalg.solve(Rt, z)
        return 0.5 * ll

    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def _unconstrained(p):
        # 无约束参数化：s=a+b 严格小于 1，t 为 a 的占比，规避边界不收敛
        s = _sigmoid(p[0])
        t = _sigmoid(p[1])
        return dcc_nll((s * t, s * (1.0 - t)))

    opt = minimize(_unconstrained, [3.5, -3.4], method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 3000})
    s = _sigmoid(opt.x[0])
    t = _sigmoid(opt.x[1])
    a, b = s * t, s * (1.0 - t)
    Q = Qbar.copy()
    Rt = np.zeros((T, n, n))
    H = np.zeros((T, n, n))
    for t in range(T):
        if t > 0:
            Q = (1 - a - b) * Qbar + a * np.outer(Z[t - 1], Z[t - 1]) + b * Q
        d = np.sqrt(np.diag(Q))
        Rt[t] = Q / np.outer(d, d)
        H[t] = np.diag(D[t]) @ Rt[t] @ np.diag(D[t])
    return {"a": float(a), "b": float(b), "Rt": Rt, "H": H, "D": D,
            "Qbar": Qbar, "columns": cols, "index": R.index,
            "univariate": uni, "success": bool(opt.success)}
