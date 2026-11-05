# -*- coding: utf-8 -*-
"""交易成本模型：显性成本（佣金/印花税/过户费）、滑点、冲击成本。"""
import numpy as np
import pandas as pd


# 默认 A 股费率（单边，bps）
DEFAULT = {
    "commission_bps": 2.5,   # 佣金（双边，万2.5）
    "stamp_bps": 5.0,        # 印花税（卖出单边，0.05%）
    "transfer_bps": 0.1,     # 过户费（双边，0.001%）
}


def round_trip_cost_bps(commission_bps=2.5, stamp_bps=5.0, transfer_bps=0.1):
    """一次完整买卖（买+卖）的单边合计成本（bps）。"""
    buy = commission_bps + transfer_bps
    sell = commission_bps + transfer_bps + stamp_bps
    return buy + sell


def impact_cost_bps(participation, coef_bps=50, model="sqrt"):
    """
    冲击成本（bps）。
    participation: 参与率（订单金额 / 当日成交量金额 ADV）。
    coef_bps: 参与率=100% 时的冲击 bps（理论标定）；默认 50，
              即参与率 1% 约 5bps、10% 约 16bps、20% 约 22bps。
    model: 'linear' 线性；'sqrt' 平方根（Almgren-Chriss 常用）。
    """
    participation = np.maximum(participation, 1e-9)
    if model == "linear":
        return coef_bps * participation
    return coef_bps * np.sqrt(participation)


def total_cost_bps(turnover, participation=None,
                   commission_bps=2.5, stamp_bps=5.0, transfer_bps=0.1,
                   impact_coef_bps=50, impact_model="sqrt"):
    """
    给定双边换手率（L1）和参与率，估算总成本（bps）。
    turnover: 双边换手率（买+卖）。
    """
    explicit = round_trip_cost_bps(commission_bps, stamp_bps, transfer_bps)
    # 显性成本按双边换手折算（round_trip 对应 turnover=2）
    explicit_part = explicit * turnover / 2.0
    impact_part = 0
    if participation is not None:
        impact_part = impact_cost_bps(participation, impact_coef_bps,
                                      impact_model)
    return explicit_part + impact_part
