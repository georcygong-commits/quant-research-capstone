# -*- coding: utf-8 -*-
"""
高频/Tick 回测（M5.3）：
逐 tick 驱动、订单延迟建模、限价单队列位置与做市策略回测。
本地无真实逐笔数据，这里用合成 tick（几何随机游走）演示机制，
notebook 中明确标注为合成数据。
"""
import numpy as np
import pandas as pd


def simulate_ticks(n=3000, p0=100.0, sigma=0.0006, spread=0.02, seed=42):
    """生成合成 tick：中间价随机游走 + 固定买卖价差。"""
    rng = np.random.default_rng(seed)
    mid = p0 * np.exp(np.cumsum(rng.normal(0, sigma, n)))
    return pd.DataFrame({"time": np.arange(n), "mid": mid,
                         "bid": mid - spread / 2, "ask": mid + spread / 2})


class LatencyModel:
    """订单固定延迟（tick 数）：提交后 delay 个 tick 才到达撮合。"""
    def __init__(self, delay_ticks=5):
        self.delay = delay_ticks

    def arrive_at(self, submit_idx):
        return submit_idx + self.delay


class QueuePosition:
    """
    限价单队列位置：新挂单排在队尾，市场成交逐量消耗队列，
    排到自己之前不会成交。
    """
    def __init__(self):
        self.ahead = 0

    def join(self, qty_ahead):
        self.ahead = int(qty_ahead)

    def consume(self, traded_qty):
        self.ahead = max(0, self.ahead - int(traded_qty))

    def reached(self):
        return self.ahead <= 0


def market_making_backtest(ticks, half_spread=0.01, latency=5,
                           max_inventory=100, rng_seed=42):
    """
    简单做市回测：
      每 tick 在 mid±half_spread 挂双边报价，订单延迟 latency 个 tick 生效，
      用到达时的对手价判断成交；库存封顶，统计做市盈亏与库存路径。
    返回库存、现金、盈亏序列。
    """
    rng = np.random.default_rng(rng_seed)
    n = len(ticks)
    mid = ticks["mid"].values
    inventory = 0
    cash = 0.0
    pnl, inv_path = [], []
    pending = []                      # (arrive_idx, side, price)
    for t in range(n):
        # 到期订单撮合
        still = []
        for arrive, side, px in pending:
            if t >= arrive:
                if side == "BUY" and mid[t] <= px and inventory < max_inventory:
                    inventory += 1; cash -= px
                elif side == "SELL" and mid[t] >= px and inventory > -max_inventory:
                    inventory -= 1; cash += px
            else:
                still.append((arrive, side, px))
        pending = still
        # 新挂双边单（带延迟）
        pending.append((t + latency, "BUY", mid[t] - half_spread))
        pending.append((t + latency, "SELL", mid[t] + half_spread))
        inv_path.append(inventory)
        pnl.append(cash + inventory * mid[t])
    out = pd.DataFrame({"time": ticks["time"], "inventory": inv_path, "pnl": pnl})
    return {"path": out, "final_pnl": float(out["pnl"].iloc[-1]),
            "max_abs_inventory": int(np.max(np.abs(inv_path))),
            "n_ticks": n}
