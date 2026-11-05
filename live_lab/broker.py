# -*- coding: utf-8 -*-
"""交易接口统一抽象与模拟实现：隔离策略与柜台。"""
from abc import ABC, abstractmethod
import pandas as pd
from live_lab.orders import Order


class BaseBroker(ABC):
    """统一交易接口：策略只依赖这些方法，底层可换券商/模拟。"""

    @abstractmethod
    def connect(self):
        ...

    @abstractmethod
    def place_order(self, order: Order):
        ...

    @abstractmethod
    def cancel_order(self, order_id):
        ...

    @abstractmethod
    def query_account(self):
        ...

    @abstractmethod
    def query_position(self):
        ...


class SimBroker(BaseBroker):
    """仿真/模拟接口：内部维护账户，立即按给定价格成交。"""

    def __init__(self, cash=1_000_000):
        self.cash = cash
        self.position = {}
        self.connected = False
        self.order_seq = 0

    def connect(self):
        self.connected = True
        return True

    def place_order(self, order: Order, fill_price=None):
        assert self.connected
        self.order_seq += 1
        px = fill_price if fill_price is not None else order.price
        sign = 1 if order.side == "BUY" else -1
        self.position[order.code] = (self.position.get(order.code, 0)
                                     + sign * order.quantity)
        self.cash -= sign * order.quantity * px
        order.filled = order.quantity
        order.status = "FILLED"
        return self.order_seq

    def cancel_order(self, order_id):
        return True

    def query_account(self):
        return {"cash": self.cash}

    def query_position(self):
        return dict(self.position)
