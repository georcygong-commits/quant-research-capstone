# -*- coding: utf-8 -*-
"""订单类型与智能下单：TWAP、VWAP、POV 拆单。"""
import numpy as np
import pandas as pd


# 订单状态
STATUS = ["NEW", "SUBMITTED", "PARTIAL", "FILLED", "CANCELLED", "REJECTED"]


class Order:
    def __init__(self, code, quantity, side, order_type="LIMIT",
                 price=None):
        self.code = code
        self.quantity = quantity
        self.side = side          # BUY / SELL
        self.type = order_type    # MARKET / LIMIT
        self.price = price
        self.filled = 0
        self.status = "NEW"

    def remaining(self):
        return self.quantity - self.filled


def twap_schedule(total_qty, n_slices):
    """TWAP：把订单均匀拆到 n 个时间片。"""
    base = total_qty / n_slices
    sched = [base] * n_slices
    # 处理取整
    sched[-1] = total_qty - sum(sched[:-1])
    return np.array(sched)


def vwap_schedule(total_qty, volume_profile):
    """VWAP：按历史成交量分布（各时段成交占比）分配。"""
    profile = np.asarray(volume_profile, dtype=float)
    profile = profile / profile.sum()
    return total_qty * profile


def pov_schedule(market_volume, target_pov, total_qty):
    """POV：按目标参与率跟随市场成交量下单，直到完成。"""
    sched = []
    remaining = total_qty
    for v in market_volume:
        q = min(target_pov * v, remaining)
        sched.append(q); remaining -= q
        if remaining <= 0:
            break
    return np.array(sched)
