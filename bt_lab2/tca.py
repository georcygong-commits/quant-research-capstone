# -*- coding: utf-8 -*-
"""
执行质量评估 TCA（M5.5）：
到达价、VWAP、执行落差（Implementation Shortfall, Perold 1988）
与滑点分解（延迟成本、市场冲击、机会成本）。
"""
import numpy as np
import pandas as pd


def vwap(prices, volumes):
    """成交量加权平均价。"""
    p = np.asarray(prices, float); v = np.asarray(volumes, float)
    return float(np.sum(p * v) / np.sum(v)) if v.sum() > 0 else np.nan


def implementation_shortfall(side, decision_px, arrival_px, fill_px, filled_qty,
                             total_qty, final_px):
    """
    执行落差（IS）：纸面组合（按决策价全成交）与实际执行的成本差，
    以成本金额（正=跑输纸面）和 bps 表示，并分解三项。
      延迟成本：决策价 -> 到达价（下单前漂移），按全部数量；
      市场冲击：到达价 -> 实际成交价，按成交数量；
      机会成本：未成交部分按最终价相对决策价的错过成本。
    """
    sign = 1.0 if side == "BUY" else -1.0
    delay = sign * (arrival_px - decision_px) * total_qty
    impact = sign * (fill_px - arrival_px) * filled_qty
    unfilled = total_qty - filled_qty
    opportunity = sign * (final_px - decision_px) * unfilled
    total = delay + impact + opportunity
    paper_notional = decision_px * total_qty
    return {"delay_cost": float(delay), "impact_cost": float(impact),
            "opportunity_cost": float(opportunity),
            "total_is": float(total),
            "is_bps": float(total / paper_notional * 10000),
            "unfilled_qty": float(unfilled)}


def slippage_vs_arrival(side, arrival_px, fill_px):
    """相对到达价的滑点（bps）。"""
    sign = 1.0 if side == "BUY" else -1.0
    return float(sign * (fill_px - arrival_px) / arrival_px * 10000)


def execution_quality_table(side, arrival_px, fill_px, mkt_vwap):
    """对比成交价相对到达价与市场 VWAP 的执行质量（bps，正=更差）。"""
    sign = 1.0 if side == "BUY" else -1.0
    return {"vs_arrival_bps": float(sign * (fill_px - arrival_px) / arrival_px * 10000),
            "vs_vwap_bps": float(sign * (fill_px - mkt_vwap) / mkt_vwap * 10000)}
