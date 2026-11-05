# -*- coding: utf-8 -*-
"""
基本面与事件因子。
基本面用 PE/PB/市值构造价值、估值分位与基本面变化；
事件因子用价量代理（异常放量、跳空、异动），真实事件需公告/研报数据。
"""
import numpy as np


def build_bf_factors(close, open_, volume, mkt, pe, pb):
    f = {}
    ret = close.pct_change()

    # ---------- 价值 / 估值 ----------
    ep = (1.0 / pe).clip(lower=0)
    bp = 1.0 / pb
    f["EP 盈利收益率"] = ep
    f["BP 账面市值比"] = bp
    f["价值综合"] = (ep + bp) / 2
    # PE 历史分位（越低越便宜，取负向）
    pe_rank = pe.rolling(250, min_periods=60).rank(pct=True)
    f["估值低分位"] = -pe_rank
    # 基本面变化：盈利收益率改善
    f["EP 改善20"] = ep.diff(20)
    f["BP 改善20"] = bp.diff(20)

    # ---------- 规模 ----------
    f["A股市值规模"] = -np.log(mkt)

    # ---------- 事件（价量代理）----------
    f["异常放量"] = -(volume / volume.rolling(20).mean())
    f["事件跳空"] = -(open_ / close.shift(1) - 1)
    # 异动幅度（大波动后反转方向）
    f["异动反转"] = -ret.rolling(5).max()
    return f
