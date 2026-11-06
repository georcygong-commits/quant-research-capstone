# -*- coding: utf-8 -*-
"""
模拟账户（paper broker）：
以份额 + 现金方式记账，按目标权重调仓并扣除双边交易成本，
每日按收盘价 mark-to-market 计算总资产与净值。
不做整手取整，使模拟净值与 capstone 向量化回测净值仅差浮点级误差。
"""
import numpy as np
import pandas as pd

from .signal_engine import COST_RATE


class PaperBroker:
    def __init__(self, init_cash=1_000_000.0, columns=None, cost_rate=COST_RATE):
        self.init_cash = float(init_cash)
        self.cash = float(init_cash)
        self.cost_rate = cost_rate
        if columns is None:
            self.shares = pd.Series(dtype=float)
        else:
            self.shares = pd.Series(0.0, index=columns, dtype=float)

    def total_value(self, prices):
        """按给定收盘价 mark-to-market 的总资产。"""
        p = prices.reindex(self.shares.index)
        return float(self.cash + (self.shares * p).sum())

    def nav(self, prices):
        """单位净值（相对初始资金，初始为 1）。"""
        return self.total_value(prices) / self.init_cash

    def current_weights(self, prices):
        """当前实际持仓权重。"""
        v = self.total_value(prices)
        p = prices.reindex(self.shares.index)
        if v <= 0:
            return pd.Series(0.0, index=self.shares.index)
        return (self.shares * p / v).fillna(0)

    def rebalance(self, target_weights, prices):
        """
        按目标权重调仓（以当日收盘价成交）：
          目标份额 = 目标权重 * 调仓前总资产 / 价格
          成本 = 双边换手金额 * cost_rate
        返回 dict（成本、双边换手权重、调仓前后总资产）。
        """
        target_weights = target_weights.reindex(self.shares.index).fillna(0)
        prices = prices.reindex(self.shares.index)
        v_before = self.total_value(prices)

        # 第一次估算换手与成本，据此预留现金，避免成本造成负现金/隐形杠杆
        target0 = (target_weights * v_before / prices).replace(
            [np.inf, -np.inf], np.nan).fillna(0.0)
        cost0 = float(((target0 - self.shares).abs() * prices).sum()) \
            * self.cost_rate
        investable = v_before - cost0
        target_shares = (target_weights * investable / prices).replace(
            [np.inf, -np.inf], np.nan).fillna(0.0)
        d_shares = target_shares - self.shares
        turnover_notional = float((d_shares.abs() * prices).sum())
        turnover_weight = turnover_notional / v_before if v_before else 0.0
        cost = turnover_notional * self.cost_rate

        # 现金：买入为负、卖出为正，再扣除成本
        self.cash += float((-d_shares * prices).sum()) - cost
        self.shares = target_shares

        v_after = self.total_value(prices)
        return {
            "cost": cost,
            "cost_weight": cost / v_before if v_before else 0.0,
            "turnover_weight": turnover_weight,
            "value_before": v_before,
            "value_after": v_after,
        }
