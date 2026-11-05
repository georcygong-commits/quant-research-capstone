# -*- coding: utf-8 -*-
"""
期权定价与 Greeks（M7.2）：
Black-Scholes 解析定价、希腊字母、CRR 二叉树、隐含波动率反解。
"""
import numpy as np
from scipy.stats import norm


def _d1(S, K, r, sigma, T):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def bs_price(S, K, r, sigma, T, option="call"):
    """Black-Scholes 欧式期权价格。"""
    d1 = _d1(S, K, r, sigma, T); d2 = d1 - sigma * np.sqrt(T)
    if option == "call":
        return float(S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2))
    return float(K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def bs_greeks(S, K, r, sigma, T, option="call"):
    """返回 delta/gamma/vega/theta（vega、theta 按常见口径）。"""
    d1 = _d1(S, K, r, sigma, T); d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1) if option == "call" else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    theta = (-(S * norm.pdf(d1) * sigma) / (2*np.sqrt(T))
             - r*K*np.exp(-r*T)*norm.cdf(d2)) / 365
    return {"delta": float(delta), "gamma": float(gamma),
            "vega": float(vega), "theta": float(theta)}


def binomial_price(S, K, r, sigma, T, n=100, option="call"):
    """CRR 二叉树：风险中性概率下倒推定价。"""
    dt = T / n; u = np.exp(sigma*np.sqrt(dt)); d = 1/u
    p = (np.exp(r*dt) - d) / (u - d)
    prices = S * u**np.arange(n+1) * d**(n-np.arange(n+1))
    vals = np.maximum(prices-K, 0) if option == "call" else np.maximum(K-prices, 0)
    for i in range(n, 0, -1):
        vals = np.exp(-r*dt) * (p*vals[1:i+1] + (1-p)*vals[0:i])
    return float(vals[0])


def implied_vol(price, S, K, r, T, option="call"):
    """二分法反解隐含波动率（稳健，避免牛顿法初值敏感）。"""
    lo, hi = 1e-4, 3.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if bs_price(S, K, r, mid, T, option) > price:
            hi = mid
        else:
            lo = mid
    return float((lo + hi) / 2)
