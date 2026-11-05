# -*- coding: utf-8 -*-
"""
规模容量曲线（M5.9）：
随管理规模上升，参与率提高、市场冲击非线性侵蚀收益，
绘制 AUM-净收益/净 Sharpe 曲线并求盈亏平衡容量。
"""
import numpy as np
import pandas as pd


def annual_impact_cost(aum, turnover, adv_annual, k):
    """
    年化冲击成本（占 AUM）：
      参与率 = AUM×换手 / 市场年可承接量；
      单次冲击 = k×参与率；年化成本 = 单次冲击×换手。
    """
    participation = aum * turnover / adv_annual
    return k * participation * turnover


def capacity_curve(aum_grid, gross_alpha, turnover, adv_annual, k=0.1,
                   fixed_cost=0.0, gross_sharpe=None, ann_vol=None):
    """不同 AUM 下的净 alpha、冲击成本与（可选）净 Sharpe。"""
    rows = []
    for a in aum_grid:
        impact = annual_impact_cost(a, turnover, adv_annual, k)
        net = gross_alpha - fixed_cost - impact
        row = {"aum": a, "impact_cost": impact, "net_alpha": net}
        if gross_sharpe is not None and gross_alpha:
            row["net_sharpe"] = gross_sharpe * net / gross_alpha
        rows.append(row)
    return pd.DataFrame(rows)


def breakeven_capacity(gross_alpha, turnover, adv_annual, k=0.1,
                       fixed_cost=0.0):
    """净 alpha 降到 0 的盈亏平衡规模（解析解）。"""
    denom = k * turnover ** 2 / adv_annual
    if denom <= 0:
        return np.inf
    return float((gross_alpha - fixed_cost) / denom)


def capacity_at_target(gross_alpha, turnover, adv_annual, k=0.1,
                       fixed_cost=0.0, target_net=0.0):
    """净 alpha 恰好等于目标值时的规模。"""
    denom = k * turnover ** 2 / adv_annual
    if denom <= 0:
        return np.inf
    return float((gross_alpha - fixed_cost - target_net) / denom)
