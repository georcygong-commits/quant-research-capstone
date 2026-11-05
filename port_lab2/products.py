# -*- coding: utf-8 -*-
"""
主流产品形态（M4.7）：
指数增强（跟踪误差 TE、超额收益、信息比率 IR）与市场中性（beta/美元中性、
股指期货对冲手数、基差）。
"""
import numpy as np
import pandas as pd
import cvxpy as cp


def tracking_error(w, Sigma, w_bench, ann=242):
    """年化跟踪误差：sqrt((w-wb)' Sigma (w-wb))。"""
    d = np.asarray(w, float) - np.asarray(w_bench, float)
    return float(np.sqrt(d @ np.asarray(Sigma) @ d) * np.sqrt(ann))


def excess_return(w, mu, w_bench, ann=242):
    """年化超额收益（相对基准）。"""
    d = np.asarray(w, float) - np.asarray(w_bench, float)
    return float(d @ np.asarray(mu) * ann)


def enhanced_index_portfolio(mu, Sigma, w_bench, te_max=0.06, gamma=4.0):
    """
    指数增强：在年化跟踪误差 <= te_max 等约束下最大化主动效用
    （alpha - 风险），保持 fully invested、long-only。
    """
    mu_v = np.asarray(mu, float)
    S_v = np.asarray(Sigma, float)
    n = len(mu_v)
    w = cp.Variable(n)
    d = w - np.asarray(w_bench)
    te_var = cp.quad_form(d, cp.psd_wrap(S_v)) * 242      # 年化 TE 方差（避免 sqrt 违反 DCP）
    obj = cp.Maximize((mu_v @ w) * 242
                      - gamma * cp.quad_form(w, cp.psd_wrap(S_v)) * 242)
    prob = cp.Problem(obj, [cp.sum(w) == 1, w >= 0, te_var <= te_max ** 2])
    prob.solve(solver=cp.CLARABEL)
    idx = mu.index if isinstance(mu, pd.Series) else None
    return pd.Series(w.value, index=idx)


def market_neutral_portfolio(mu, Sigma, betas, gross_limit=1.0):
    """
    市场中性多空组合：允许负权重，约束
      beta 中性 w'beta = 0；美元中性 sum(w) = 0；总敞口 ||w||_1 <= gross_limit。
    最大化 mu'w - gamma w'Sigma w。gross 约束防止无约束下杠杆发散。
    """
    mu_v = np.asarray(mu, float)
    S_v = np.asarray(Sigma, float)
    n = len(mu_v)
    w = cp.Variable(n)
    obj = cp.Maximize(mu_v @ w - 4.0 * cp.quad_form(w, cp.psd_wrap(S_v)))
    prob = cp.Problem(obj, [w @ np.asarray(betas) == 0, cp.sum(w) == 0,
                            cp.norm(w, 1) <= gross_limit])
    prob.solve(solver=cp.CLARABEL)
    idx = mu.index if isinstance(mu, pd.Series) else None
    return pd.Series(w.value, index=idx)


def index_hedge_contracts(beta, portfolio_value, index_price, multiplier=300):
    """
    股指期货对冲所需合约数（向下取整）：
      N = beta * 组合市值 / (指数点位 * 合约乘数)
    沪深300 IF 乘数 300，中证500 IC 乘数 200，上证50 IH 乘数 300。
    """
    nominal = beta * portfolio_value
    return int(np.floor(nominal / (index_price * multiplier)))


def gross_net_exposure(w):
    """多空组合的总敞口（gross）与净敞口（net）。"""
    w = np.asarray(w, float)
    return {"gross": float(np.abs(w).sum()), "net": float(w.sum()),
            "long": float(w[w > 0].sum()), "short": float(w[w < 0].sum())}
