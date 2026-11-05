# -*- coding: utf-8 -*-
"""纸面交易（paper trading）：模拟账户、订单、成交、持仓。"""
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class Order:
    code: str
    target_weight: float
    filled: bool = False


class PaperBroker:
    """
    纸面券商：给定目标权重，按当前价格"成交"，维护现金/持仓/净值。
    用于实盘前的模拟盘验证（不接真实资金）。
    """

    def __init__(self, cash=1_000_000, cost_bps=10):
        self.cash = cash
        self.positions = {}      # code -> 股数
        self.cost_bps = cost_bps
        self.trades = []

    def rebalance(self, target_weights, prices):
        """按目标权重（当前价）调仓，返回成交记录。"""
        nav = self.net_value(prices)
        before = dict(self.positions)
        self.positions = {}
        turnover = 0
        for code, w in target_weights.items():
            px = prices.get(code)
            if px is None or px <= 0:
                continue
            shares = nav * w / px
            self.positions[code] = shares
            old = before.get(code, 0)
            turnover += abs(shares - old) * px
        cost = turnover * self.cost_bps / 10000
        self.cash = nav - sum(self.positions[c] * prices[c]
                              for c in self.positions) - cost
        self.trades.append({"turnover": turnover, "cost": cost})
        return self.trades[-1]

    def net_value(self, prices):
        v = self.cash
        for c, s in self.positions.items():
            px = prices.get(c)
            if px:
                v += s * px
        return v
