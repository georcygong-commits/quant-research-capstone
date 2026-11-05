# -*- coding: utf-8 -*-
"""因子风险模型（Barra 式）：因子暴露 -> 因子收益 -> 因子协方差 -> 特质风险。"""
import numpy as np
import pandas as pd


def estimate_factor_returns(returns, exposures, weights=None):
    """
    每个截面做加权回归 r = B f + eps，估计因子收益。
    returns: 收益宽表；exposures: {因子名: 暴露宽表}。
    返回：因子收益宽表（日期×因子），残差宽表。
    """
    names = list(exposures.keys())
    fr = {}; res = {}
    for dt in returns.index:
        y = returns.loc[dt]
        X = pd.DataFrame({k: exposures[k].loc[dt] for k in names})
        ok = y.notna() & X.notna().all(axis=1)
        if ok.sum() < len(names) + 5:
            continue
        Xv = X[ok].values; yv = y[ok].values
        if weights is not None:
            w = np.sqrt(weights.loc[dt, ok].values)
            Xv = Xv * w[:, None]; yv = yv * w
        f, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
        fr[dt] = f
        e = pd.Series(np.nan, index=X.index)
        e[ok] = yv - X[ok].values @ f
        res[dt] = e
    return pd.DataFrame(fr, index=names).T, pd.DataFrame(res).T


def factor_risk_model(returns, exposures, ann=242):
    """
    构建因子风险模型：
    返回因子协方差 Sigma_f、特质风险 diag(D)、暴露面板。
    """
    fr, res = estimate_factor_returns(returns, exposures)
    Sigma_f = fr.cov() * ann
    spec = (res.var() * ann).fillna(0)
    return {"factor_returns": fr, "Sigma_f": Sigma_f,
            "specific_risk": spec, "residual": res}


def reconstruct_cov(model, exposures, dt):
    """用因子模型重建某一天的协方差：Σ = B Σ_f B^T + D。"""
    B = pd.DataFrame({k: exposures[k].loc[dt] for k in model["Sigma_f"].index})
    B = B.fillna(0)
    Sf = model["Sigma_f"].values
    D = np.diag(model["specific_risk"].reindex(B.index).fillna(0).values)
    return B.values @ Sf @ B.values.T + D, B
