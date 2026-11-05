# -*- coding: utf-8 -*-
"""
整手/整数规划与成本感知优化（M4.6）：
A 股最小交易单位为 1 手 = 100 股，真实下单不能买分数股。
实现目标权重的整手圆整、cvxpy 混合整数规划（MIP）、含换手成本的连续优化。
"""
import numpy as np
import pandas as pd
import cvxpy as cp


def round_to_lots(weights, prices, capital, lot=100):
    """
    把目标权重圆整为整手股数：
      目标金额 -> 目标股数 -> 向下取整到 lot 的倍数。
    返回整手权重、实际占用资金、现金残留比例。
    """
    w = np.asarray(weights, float)
    p = np.asarray(prices, float)
    target_shares = w * capital / p
    lots = np.floor(target_shares / lot)
    shares = lots * lot
    invested = shares * p
    w_real = invested / capital
    return {"shares": pd.Series(shares, index=weights.index),
            "weights": pd.Series(w_real, index=weights.index),
            "invested": float(invested.sum()),
            "cash_residual": float(1.0 - invested.sum() / capital),
            "n_held": int((shares > 0).sum())}


def mip_portfolio(mu, Sigma, prices, capital, lot=100, gamma=5.0,
                  cost_bps=0.0, prev_shares=None):
    """
    混合整数规划（MIP，资本预算/背包模型）：决策变量为「手数」（非负整数），
    在预算约束下最大化期望收益（扣除单边买入成本）。
    用 scipy.optimize.milp（HiGHS 分支定界）求解。cvxpy 1.7 已移除 ECOS_BB，
    开源整数 solver 不支持二次锥，故二次风险由连续优化 + round_to_lots 承担，
    这里求解线性整数规划；Sigma/gamma 保留接口、不进入线性目标。
    """
    from scipy.optimize import milp, LinearConstraint, Bounds
    mu_v = np.asarray(mu, float)
    p_v = np.asarray(prices, float)
    n = len(mu_v)
    cost_per_lot = lot * p_v
    net_mu = mu_v - cost_bps / 1e4                  # 扣单边买入成本
    c = -net_mu * cost_per_lot                      # milp 为最小化
    lc = LinearConstraint(cost_per_lot.reshape(1, n), -np.inf, capital)
    ub = np.floor(capital / cost_per_lot)
    res = milp(c=c, constraints=lc, integrality=np.ones(n),
               bounds=Bounds(np.zeros(n), ub))
    hv = np.round(res.x).astype(int)
    invested = hv * cost_per_lot
    idx = mu.index if isinstance(mu, pd.Series) else None
    return {"lots": pd.Series(hv, index=idx),
            "shares": pd.Series(hv * lot, index=idx),
            "weights": pd.Series(invested / capital, index=idx),
            "status": "optimal" if res.success else str(res.message),
            "objective": float(-res.fun)}


def cost_aware_weights(mu, Sigma, prev_w, cost_bps=20.0, gamma=5.0):
    """
    成本感知连续优化：在均值-方差目标中加入换手惩罚
        c * ||w - w_prev||_1, c = cost_bps/10000
    成本越高，最优组合越靠近原持仓、换手越低。
    """
    mu_v = np.asarray(mu, float)
    S_v = np.asarray(Sigma, float)
    n = len(mu_v)
    w = cp.Variable(n)
    c = cost_bps / 1e4
    obj = cp.Maximize(mu_v @ w - gamma * cp.quad_form(w, cp.psd_wrap(S_v))
                      - c * cp.norm(w - np.asarray(prev_w), 1))
    prob = cp.Problem(obj, [cp.sum(w) == 1, w >= 0])
    prob.solve(solver=cp.CLARABEL)
    idx = mu.index if isinstance(mu, pd.Series) else None
    return pd.Series(w.value, index=idx)


def turnover(w_new, w_old):
    """单边换手率（权重差绝对值之和）。"""
    return float(np.sum(np.abs(np.asarray(w_new) - np.asarray(w_old))))
