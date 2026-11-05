# -*- coding: utf-8 -*-
"""限价单簿（LOB）：连续竞价撮合、盘口、深度。教学版价格-时间优先。"""


class OrderBook:
    """维护买卖两侧价格档位，支持限价单撮合与挂单、撤单、盘口查询。"""

    def __init__(self):
        self.bids = {}   # price -> quantity
        self.asks = {}
        self.trades = []

    # ---------- 盘口 ----------
    def best_bid(self):
        return max(self.bids) if self.bids else None

    def best_ask(self):
        return min(self.asks) if self.asks else None

    def mid_price(self):
        b, a = self.best_bid(), self.best_ask()
        if b is not None and a is not None:
            return (b + a) / 2
        return b or a

    def quoted_spread(self):
        b, a = self.best_bid(), self.best_ask()
        if b is None or a is None:
            return None
        return a - b

    def depth(self, side, n_levels=5):
        book = self.bids if side == "bid" else self.asks
        prices = sorted(book, reverse=(side == "bid"))[:n_levels]
        return [(p, book[p]) for p in prices]

    # ---------- 撮合 ----------
    def limit_order(self, side, price, qty):
        """提交限价单：先可成交部分撮合，剩余挂单。返回成交记录。"""
        remaining = qty
        if side == "BUY":
            while remaining and self.asks:
                a = self.best_ask()
                if a > price:
                    break
                take = min(remaining, self.asks[a])
                self.trades.append((a, take, "BUY"))
                remaining -= take
                self.asks[a] -= take
                if self.asks[a] == 0:
                    del self.asks[a]
            if remaining:
                self.bids[price] = self.bids.get(price, 0) + remaining
        else:
            while remaining and self.bids:
                b = self.best_bid()
                if b < price:
                    break
                take = min(remaining, self.bids[b])
                self.trades.append((b, take, "SELL"))
                remaining -= take
                self.bids[b] -= take
                if self.bids[b] == 0:
                    del self.bids[b]
            if remaining:
                self.asks[price] = self.asks.get(price, 0) + remaining
        return self.trades[-1] if self.trades else None

    def cancel(self, side, price):
        book = self.bids if side == "bid" else self.asks
        if price in book:
            del book[price]
            return True
        return False
