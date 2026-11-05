# -*- coding: utf-8 -*-
"""买卖价差与深度指标：报价价差、有效价差、实现价差。"""
import numpy as np


def quoted_spread(best_bid, best_ask):
    return best_ask - best_bid


def relative_spread(best_bid, best_ask):
    mid = (best_bid + best_ask) / 2
    return (best_ask - best_bid) / mid


def effective_spread(trade_price, mid, side):
    """
    有效价差：实际成交价相对中间价的偏离。
    side: +1 买方主动成交，-1 卖方主动成交。
    """
    return 2 * side * (trade_price - mid)


def relative_effective_spread(trade_price, mid, side):
    return effective_spread(trade_price, mid, side) / mid


def realized_spread(trade_price, future_mid, side):
    """实现价差：成交价相对成交后未来中间价的偏离（扣除逆向选择部分）。"""
    return 2 * side * (trade_price - future_mid)


def book_depth(book, n_levels=5, side="bid"):
    """前 n 档累计挂单量。"""
    prices = sorted(book, reverse=(side == "bid"))[:n_levels]
    return sum(book[p] for p in prices)
