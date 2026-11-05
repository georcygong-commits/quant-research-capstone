# -*- coding: utf-8 -*-
"""
盘口撮合与流动性约束（M5.2）：
限价订单簿（LOB）、市价单吃单与部分成交、涨跌停/停牌/参与率约束，
以及线性市场冲击（滑点）模型。
"""
import numpy as np
import pandas as pd


class OrderBook:
    """简单限价订单簿：价格 -> 累计挂单量。"""
    def __init__(self):
        self.bids = {}
        self.asks = {}

    def add_limit(self, side, price, qty):
        book = self.bids if side == "BUY" else self.asks
        book[price] = book.get(price, 0) + qty

    def cancel(self, side, price, qty):
        book = self.bids if side == "BUY" else self.asks
        book[price] = max(0, book.get(price, 0) - qty)

    def best_bid(self):
        return max(self.bids) if self.bids else None

    def best_ask(self):
        return min(self.asks) if self.asks else None

    def mid(self):
        b, a = self.best_bid(), self.best_ask()
        return None if b is None or a is None else 0.5 * (b + a)

    def spread(self):
        b, a = self.best_bid(), self.best_ask()
        return None if b is None or a is None else a - b

    def market_order(self, side, qty):
        """逐档吃单，返回 (成交列表[(价,量)], 未成交量)。"""
        fills, remaining = [], qty
        if side == "BUY":
            levels = sorted(self.asks)
            for px in levels:
                take = min(self.asks[px], remaining)
                if take:
                    fills.append((px, take)); remaining -= take
                if remaining == 0:
                    break
        else:
            levels = sorted(self.bids, reverse=True)
            for px in levels:
                take = min(self.bids[px], remaining)
                if take:
                    fills.append((px, take)); remaining -= take
                if remaining == 0:
                    break
        return fills, remaining


def limit_up_down(prev_close, price, threshold=0.099):
    """返回涨停/跌停标志（相对昨收）。"""
    chg = price / prev_close - 1.0
    return {"limit_up": bool(chg >= threshold), "limit_down": bool(chg <= -threshold),
            "change": float(chg)}


def is_halted(bar):
    """成交量为 0 或价格缺失视为停牌。"""
    return bool(bar.get("volume", 0) <= 0 or np.isnan(bar.get("close", np.nan)))


def max_participation_qty(volume, participation=0.1):
    """给定参与率下的最大可成交数量。"""
    return float(volume * participation)


def market_impact(qty, adv, k=0.1):
    """
    线性市场冲击（占价比）：k * (qty/ADV)。
    大单相对日均成交越集中，冲击成本越高。
    """
    if adv <= 0:
        return np.inf
    return float(k * qty / adv)


def execute_with_constraints(side, qty, bar, prev_close, participation=0.1,
                             impact_k=0.1):
    """
    综合流动性约束的单笔执行：
      停牌 -> 拒单；涨跌停 -> 对应方向拒单；
      参与率截断数量；计算加权成交价与冲击成本。
    返回成交结果或拒单原因。
    """
    if is_halted(bar):
        return {"status": "REJECTED", "reason": "HALTED"}
    lu = limit_up_down(prev_close, bar["open"])
    if side == "BUY" and lu["limit_up"]:
        return {"status": "REJECTED", "reason": "LIMIT_UP"}
    if side == "SELL" and lu["limit_down"]:
        return {"status": "REJECTED", "reason": "LIMIT_DOWN"}
    cap_qty = min(qty, max_participation_qty(bar["volume"], participation))
    base_px = bar["open"]
    impact = market_impact(cap_qty, bar["volume"], impact_k)
    fill_px = base_px * (1 + impact) if side == "BUY" else base_px * (1 - impact)
    return {"status": "PARTIAL" if cap_qty < qty else "FILLED",
            "filled_qty": float(cap_qty), "unfilled": float(qty - cap_qty),
            "avg_price": float(fill_px), "impact": float(impact)}
