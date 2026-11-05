# -*- coding: utf-8 -*-
"""
做市与高频（M7.5）：
做市双边报价、库存管理偏移、逆向选择成本与已实现价差。
"""
import numpy as np


def make_quote(mid, half_spread, inventory=0, skew_coef=0.0):
    """
    双边报价：中间价两侧挂单，并按库存做偏移。
    库存偏多时整体下移报价（更易卖出、减少买入），反之亦然。
    """
    skew = skew_coef * inventory
    return float(mid - half_spread - skew), float(mid + half_spread - skew)


def inventory_pnl(inventory, mid, ref):
    """库存盯市盈亏。"""
    return float(inventory * (mid - ref))


def quoted_spread(bid, ask):
    """报价价差（bps）。"""
    mid = (bid + ask) / 2
    return float((ask - bid) / mid * 10000)


def adverse_selection_cost(price_change, trade_side):
    """
    逆向选择成本（简化）：成交后价格朝成交方向继续走，
    做市商被知情交易者「占便宜」的部分。trade_side: +1 客户买入 -1 卖出。
    """
    return float(trade_side * price_change)


def effective_spread(trade_px, mid, side):
    """有效价差：实际成交价相对中间价的偏离。"""
    return float(2 * side * (trade_px - mid) / mid * 10000)
