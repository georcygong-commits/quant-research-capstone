# -*- coding: utf-8 -*-
"""
实时行情与信号（M6.1）：
行情清洗（坏值/跳变过滤）、增量滚动信号（O(1) 状态更新）、
增强交易时钟（时段判断、距收盘时间）。
"""
from collections import deque
import numpy as np


def clean_tick(price, prev_price=None, max_jump=0.1):
    """过滤非正/缺失值与相对上一 tick 跳变超阈值的可疑行情。"""
    if not np.isfinite(price) or price <= 0:
        return None
    if prev_price is not None and abs(price / prev_price - 1.0) > max_jump:
        return None
    return float(price)


class IncrementalMean:
    """定长窗口的增量均值：维护队列与和，新 tick 到达时 O(1) 更新。"""
    def __init__(self, window):
        self.window = window
        self.q = deque()
        self.total = 0.0

    def update(self, x):
        self.q.append(x); self.total += x
        if len(self.q) > self.window:
            self.total -= self.q.popleft()
        return self.value()

    def value(self):
        return self.total / len(self.q) if self.q else np.nan


class IncrementalVolatility(IncrementalMean):
    """增量已实现波动：用 Welford 式在线二阶矩（教学简化为收益平方均值）。"""
    def __init__(self, window):
        super().__init__(window)
        self.sq = 0.0
        self.sq_q = deque()

    def update(self, x):
        self.sq_q.append(x * x); self.sq += x * x
        if len(self.sq_q) > self.window:
            self.sq -= self.sq_q.popleft()
        return self.value()

    def value(self):
        return float(np.sqrt(self.sq / len(self.sq_q))) if self.sq_q else np.nan


class TradingClock:
    """交易时钟：判断是否在交易时段，计算距收盘分钟数。"""
    def __init__(self, sessions=(("09:30", "11:30"), ("13:00", "15:00"))):
        self.sessions = sessions

    @staticmethod
    def _mins(hhmm):
        h, m = hhmm.split(":")
        return int(h) * 60 + int(m)

    def in_session(self, hhmm):
        cur = self._mins(hhmm)
        return any(self._mins(a) <= cur <= self._mins(b) for a, b in self.sessions)

    def minutes_to_close(self, hhmm):
        cur = self._mins(hhmm)
        for a, b in self.sessions:
            if self._mins(a) <= cur <= self._mins(b):
                return self._mins(b) - cur
        return None
