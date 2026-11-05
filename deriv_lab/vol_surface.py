# -*- coding: utf-8 -*-
"""
波动率曲面与交易（M7.3）：
波动率微笑/偏斜拟合、曲面构建、隐含波动率与已实现波动率的波动率交易。
"""
import numpy as np


def fit_smile(moneyness, iv):
    """用二次函数拟合波动率微笑（偏斜+曲率，教学简化）。"""
    coef = np.polyfit(np.asarray(moneyness), np.asarray(iv), 2)
    return coef                      # a*x^2 + b*x + c


def smile_iv(coef, moneyness):
    return float(np.polyval(coef, moneyness))


def build_surface(strikes, tenors, iv_grid):
    """记录曲面网格（实盘用 SVI/SABR 等参数模型，这里保留网格与口径）。"""
    return {"strikes": np.asarray(strikes), "tenors": np.asarray(tenors),
            "iv": np.asarray(iv_grid)}


def vol_trade_pnl(implied_vol, realized_vol, vega_exposure):
    """
    波动率交易损益（delta 中性、近似口径）：
    做空波动率时，已实现低于隐含则获利；损益 ≈ vega 暴露 × (IV-RV)。
    """
    return float(vega_exposure * (implied_vol - realized_vol) * 100)


def variance_swap_payoff(realized_var, strike_var, vega_notional):
    """方差互换到期损益 = (已实现方差 - 行权方差) × vega 名义。"""
    return float((realized_var - strike_var) * vega_notional)
