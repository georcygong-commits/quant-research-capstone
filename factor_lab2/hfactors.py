# -*- coding: utf-8 -*-
"""
高频与微观结构因子（日频可得的代理实现）。
真实分钟/tick 因子需分钟数据，这里用 OHLC 构造日内行为与分布因子。
输入均为宽表（date × code）。
"""
import numpy as np


def build_hf_factors(open_, close, high, low, volume):
    ret = close.pct_change()
    f = {}

    # ---------- 分布特征 ----------
    f["已实现波动20"] = -ret.rolling(20).std()          # 低波动方向
    f["收益偏度60"] = -ret.rolling(60).skew()           # 负偏度
    f["收益峰度60"] = -ret.rolling(60).kurt()           # 低尾部
    f["日内振幅20"] = -((high - low) / close).rolling(20).mean()

    # ---------- 日内 / 隔夜行为 ----------
    overnight = open_ / close.shift(1) - 1              # 隔夜跳空
    intraday = close / open_ - 1                        # 日内收益
    f["隔夜跳空20"] = -overnight.rolling(20).mean()
    f["日内动量20"] = intraday.rolling(20).mean()
    # 收盘在当日区间的位置（尾盘强弱）
    loc = (close - low) / (high - low).replace(0, np.nan)
    f["尾盘位置20"] = loc.rolling(20).mean()

    # ---------- 影线 ----------
    upper = (high - np.maximum(open_, close)) / close
    lower = (np.minimum(open_, close) - low) / close
    f["上影线20"] = -upper.rolling(20).mean()
    f["下影线20"] = lower.rolling(20).mean()

    # ---------- 量价关系 ----------
    f["量价相关20"] = -ret.rolling(20).corr(volume.pct_change(fill_method=None))
    f["放量波动20"] = -(volume.rolling(20).std() /
                       volume.rolling(20).mean())

    return f
