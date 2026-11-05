# -*- coding: utf-8 -*-
"""
事件驱动回测引擎（M5.1）：
事件队列（MARKET/SIGNAL/ORDER/FILL）+ 订单/成交 + 现金持仓台账，
策略通过 on_bar 产生订单，引擎按日循环撮合并记录净值。
"""
from collections import deque
import numpy as np
import pandas as pd


# ---------------- 事件 ----------------
class Event:
    def __init__(self, etype, data=None):
        self.type = etype          # MARKET / SIGNAL / ORDER / FILL
        self.data = data

    def __repr__(self):
        return "Event(%s)" % self.type


class EventQueue:
    def __init__(self):
        self._q = deque()

    def put(self, event):
        self._q.append(event)

    def get(self):
        return self._q.popleft() if self._q else None

    def __len__(self):
        return len(self._q)


# ---------------- 订单 / 成交 ----------------
class Order:
    def __init__(self, code, quantity, side, order_type="MARKET", price=None):
        self.code = code
        self.quantity = int(quantity)      # 正数
        self.side = side                   # BUY / SELL
        self.type = order_type             # MARKET / LIMIT
        self.price = price
        self.status = "NEW"

    def __repr__(self):
        return "Order(%s %s %d %s)" % (self.side, self.code, self.quantity, self.type)


class Fill:
    def __init__(self, code, quantity, side, price, commission):
        self.code = code
        self.quantity = quantity
        self.side = side
        self.price = price
        self.commission = commission


# ---------------- 台账 ----------------
class Portfolio:
    def __init__(self, cash, commission_bps=2.0):
        self.init_cash = cash
        self.cash = cash
        self.positions = {}               # code -> 股数
        self.commission_bps = commission_bps
        self.nav = []

    def apply_fill(self, fill):
        notional = fill.price * fill.quantity
        if fill.side == "BUY":
            self.cash -= notional + fill.commission
            self.positions[fill.code] = self.positions.get(fill.code, 0) + fill.quantity
        else:
            self.cash += notional - fill.commission
            self.positions[fill.code] = self.positions.get(fill.code, 0) - fill.quantity

    def equity(self, prices):
        mv = sum(qty * prices.get(code, 0.0) for code, qty in self.positions.items())
        return self.cash + mv

    def record(self, date, prices):
        self.nav.append((date, self.equity(prices)))


# ---------------- 简单撮合 ----------------
class BasicMatcher:
    """市价单按当前 bar 开盘价全成交；限价单判断价格是否触及。"""
    def match(self, order, bar):
        if order.type == "MARKET":
            px = bar["open"]
        else:
            if order.side == "BUY" and bar["low"] <= order.price:
                px = min(bar["open"], order.price)
            elif order.side == "SELL" and bar["high"] >= order.price:
                px = max(bar["open"], order.price)
            else:
                return None
        commission = px * order.quantity * 0.0
        return Fill(order.code, order.quantity, order.side, px, commission)


# ---------------- 引擎 ----------------
class EventDrivenBacktest:
    def __init__(self, bars, strategy, init_cash=1e7, matcher=None):
        """
        bars: dict[code] = DataFrame(index=date, columns open/high/low/close/volume)
        strategy: 含 on_bar(date, bar_dict, portfolio) -> [Order]
        """
        self.bars = bars
        self.dates = sorted(set().union(*[df.index for df in bars.values()]))
        self.strategy = strategy
        self.portfolio = Portfolio(init_cash)
        self.matcher = matcher or BasicMatcher()
        self.fills = []
        self.queue = EventQueue()

    def _bar_dict(self, date):
        out = {}
        for code, df in self.bars.items():
            if date in df.index:
                row = df.loc[date]
                out[code] = {"open": row["open"], "high": row["high"],
                             "low": row["low"], "close": row["close"],
                             "volume": row["volume"]}
        return out

    def run(self):
        for date in self.dates:
            bar_dict = self._bar_dict(date)
            self.queue.put(Event("MARKET", bar_dict))
            while len(self.queue):
                ev = self.queue.get()
                if ev.type == "MARKET":
                    for order in self.strategy.on_bar(date, ev.data, self.portfolio):
                        self.queue.put(Event("ORDER", order))
                elif ev.type == "ORDER":
                    order = ev.data
                    if order.code in bar_dict:
                        fill = self.matcher.match(order, bar_dict[order.code])
                        if fill is not None:
                            self.queue.put(Event("FILL", fill))
                elif ev.type == "FILL":
                    self.portfolio.apply_fill(ev.data)
                    self.fills.append(ev.data)
            prices = {c: b["close"] for c, b in bar_dict.items()}
            self.portfolio.record(date, prices)
        return self.nav_frame()

    def nav_frame(self):
        df = pd.DataFrame(self.portfolio.nav, columns=["date", "equity"]).set_index("date")
        df["return"] = df["equity"].pct_change(fill_method=None).fillna(0)
        return df
