# -*- coding: utf-8 -*-
"""实时行情接入：交易时钟与模拟行情源（可替换为真实行情接口）。"""
import numpy as np


class MarketClock:
    """管理交易时段（A 股 9:30-11:30、13:00-15:00）。"""

    def __init__(self, slots=(("09:30", "11:30"), ("13:00", "15:00"))):
        self.slots = slots

    def is_trading_time(self, hhmm):
        return any(a <= hhmm <= b for a, b in self.slots)


class MockMarketFeed:
    """
    模拟行情源：按给定初始价，用随机游走逐拍生成 tick，
    实盘中可替换为券商/交易所行情订阅（同样的 next/订阅接口）。
    """

    def __init__(self, init_prices, n_ticks=60, seed=42, vol=0.001):
        self.prices = dict(init_prices)
        self.codes = list(init_prices)
        self.n_ticks = n_ticks
        self.vol = vol
        self.rng = np.random.default_rng(seed)
        self.t = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.t >= self.n_ticks:
            raise StopIteration
        snap = {}
        for c in self.codes:
            shock = self.rng.normal(0, self.vol)
            self.prices[c] = max(self.prices[c] * (1 + shock), 0.01)
            snap[c] = round(self.prices[c], 3)
        self.t += 1
        return {"tick": self.t, "prices": snap}
