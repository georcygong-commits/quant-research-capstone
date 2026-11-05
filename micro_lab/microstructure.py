# -*- coding: utf-8 -*-
"""
微观结构指标：Kyle's lambda（statsmodels 与 numpy 双实现）、
Amihud、VPIN、订单流不平衡 OFI、价格冲击。
"""
import numpy as np


# ---------- Kyle's lambda：价格冲击回归 ----------
def kyle_lambda_statsmodels(price_change, signed_volume):
    """ΔP = α + λ·订单流 + ε，用 statsmodels OLS，返回 (lambda, 模型)。"""
    import statsmodels.api as sm
    x = np.asarray(signed_volume, dtype=float)
    y = np.asarray(price_change, dtype=float)
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    return model.params[1], model


def kyle_lambda_numpy(price_change, signed_volume):
    """同一回归，用 numpy 最小二乘手写，返回 (lambda, intercept)。"""
    x = np.asarray(signed_volume, dtype=float)
    y = np.asarray(price_change, dtype=float)
    xbar, ybar = x.mean(), y.mean()
    slope = np.sum((x - xbar) * (y - ybar)) / np.sum((x - xbar) ** 2)
    intercept = ybar - slope * xbar
    return slope, intercept


# ---------- Amihud 非流动性 ----------
def amihud_illiquidity(returns, dollar_volume):
    """ILLIQ = mean(|r| / 成交额)，值越大越缺乏流动性。"""
    r = np.asarray(returns, dtype=float)
    dv = np.asarray(dollar_volume, dtype=float)
    dv = np.where(dv <= 0, np.nan, dv)
    return np.nanmean(np.abs(r) / dv)


# ---------- VPIN ----------
def vpin(prices, volumes, n_buckets=10):
    """
    按等成交量篮子，用价格涨跌方向拆分买卖量，
    VPIN = 平均(|买量-卖量| / 篮子总量)，衡量知情交易/毒性。
    """
    prices = np.asarray(prices, float)
    volumes = np.asarray(volumes, float)
    total = volumes.sum()
    bucket_size = total / n_buckets
    imb = []
    cur = 0.0; buy = 0.0; sell = 0.0
    direction = np.sign(np.diff(prices, prepend=prices[0]))
    for v, d in zip(volumes, direction):
        d = d if d != 0 else 1
        b = v if d > 0 else 0
        s = v if d < 0 else 0
        cur += v; buy += b; sell += s
        if cur >= bucket_size:
            imb.append(abs(buy - sell) / max(cur, 1e-12))
            cur = 0; buy = 0; sell = 0
    if cur > 0:
        imb.append(abs(buy - sell) / cur)
    return float(np.mean(imb))


# ---------- 订单流不平衡 OFI ----------
def order_flow_imbalance(prev_bid, bid, prev_bid_q, bid_q,
                         prev_ask, ask, prev_ask_q, ask_q):
    """单期 OFI：买盘挂单变化减去卖盘挂单变化（Best-level 教学版）。"""
    bid_ofi = (bid_q if bid >= prev_bid else 0) - \
              (prev_bid_q if bid <= prev_bid else 0)
    ask_ofi = (ask_q if ask <= prev_ask else 0) - \
              (prev_ask_q if ask >= prev_ask else 0)
    return bid_ofi - ask_ofi


# ---------- 价格冲击（平方根模型，与回测成本呼应）----------
def square_root_impact(participation_rate, daily_vol, coef=1.0):
    """临时冲击 ≈ coef × 波动 × sqrt(参与率)。"""
    return coef * daily_vol * np.sqrt(participation_rate)
