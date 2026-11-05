# -*- coding: utf-8 -*-
"""
因子库：批量构造量价、基本面、风格因子（方向统一为正向预测）。
输入为各字段的宽表（date × code），返回 {因子名: 因子宽表}。
"""
import numpy as np
import pandas as pd


def build_factors(close, volume, mkt, pe, pb):
    ret1 = close.pct_change()
    f = {}

    # ---------- 规模 / 价值 ----------
    f["SMB 小市值"] = -np.log(mkt)
    f["EP 盈利收益率"] = (1.0 / pe).clip(lower=0)     # 亏损公司 EP 置 0
    f["BP 账面市值比"] = 1.0 / pb

    # ---------- 反转 / 动量 ----------
    f["REV5 短期反转"] = -close.pct_change(5)
    f["REV20 中期反转"] = -close.pct_change(20)
    f["MOM120 长期动量"] = close.pct_change(120)

    # ---------- 波动率 ----------
    f["低波20"] = -ret1.rolling(20).std()
    f["低波60"] = -ret1.rolling(60).std()

    # ---------- 流动性 / 换手 ----------
    # 换手率近似：成交量 / 市值（口径近似，仅用于相对排序）
    f["低换手20"] = -np.log((volume.rolling(20).mean() / mkt).replace(0, np.nan))
    f["低成交额20"] = -np.log(volume.rolling(20).mean().replace(0, np.nan))

    # ---------- 形态 / 分布 ----------
    f["低振幅20"] = -((close.rolling(20).max() - close.rolling(20).min())
                      / close.rolling(20).mean())
    f["负偏度60"] = -ret1.rolling(60).skew()

    # ---------- 隔夜 / 日内 ----------
    # 隔夜跳空（近似：用 open 不可得时略过；这里用日收益的自相关）
    f["低量比"] = -(volume / volume.rolling(20).mean())

    return f
