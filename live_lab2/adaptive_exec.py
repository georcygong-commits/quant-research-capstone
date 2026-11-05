# -*- coding: utf-8 -*-
"""
自适应订单执行（M6.3）：
根据实时成交量/参与率反馈动态调整 slice 的 POV 执行、
订单超时/拒单重试、实时 TCA 累计。
"""
import numpy as np


class AdaptivePOV:
    """
    自适应 POV：每个时间片按目标参与率跟随市场成交量，
    实际参与率偏高时自动减速，直到完成。
    """
    def __init__(self, total_qty, target_pov=0.1):
        self.remaining = total_qty
        self.target_pov = target_pov
        self.slices = []

    def next_slice(self, market_volume, realized_pov=None):
        pov = self.target_pov
        if realized_pov is not None and realized_pov > self.target_pov:
            pov *= self.target_pov / realized_pov          # 超参与率则减速
        q = min(pov * market_volume, self.remaining)
        self.remaining -= q; self.slices.append(q)
        return float(q)

    def done(self):
        return self.remaining <= 0


class OrderManager:
    """订单生命周期管理：拒单按退避重试、超时判定。"""
    def __init__(self, max_retry=3, timeout_s=30):
        self.max_retry = max_retry
        self.timeout = timeout_s
        self.orders = {}
        self.retry_count = {}

    def submit(self, order_id, order):
        self.orders[order_id] = {"order": order, "status": "SUBMITTED",
                                 "submit_time": 0, "retries": 0}
        return order_id

    def on_reject(self, order_id):
        rec = self.orders[order_id]
        if rec["retries"] >= self.max_retry:
            rec["status"] = "DEAD"
            return False
        rec["retries"] += 1; rec["status"] = "RETRYING"
        return True

    def is_timeout(self, order_id, now):
        rec = self.orders[order_id]
        if now - rec["submit_time"] > self.timeout and rec["status"] != "FILLED":
            rec["status"] = "TIMEOUT"
            return True
        return False


class RealtimeTCA:
    """执行过程中实时累计成交量、加权均价与相对到达价滑点。"""
    def __init__(self, arrival_px):
        self.arrival = arrival_px
        self.filled_qty = 0.0
        self.notional = 0.0

    def update(self, px, qty):
        self.filled_qty += qty
        self.notional += px * qty
        return self.snapshot()

    def avg_px(self):
        return self.notional / self.filled_qty if self.filled_qty else np.nan

    def snapshot(self):
        return {"filled": self.filled_qty, "avg_px": self.avg_px(),
                "slippage_bps": (self.avg_px()/self.arrival - 1) * 10000
                if self.filled_qty else 0.0}
