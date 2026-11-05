# -*- coding: utf-8 -*-
"""
Barra 风格因子（CNE 式教学实现），用日频可得数据构造。
含 Size/Beta/Momentum/Residual Volatility/Non-linear Size/
Liquidity/BTP/EP/Leverage(占位)/Growth。
"""
import numpy as np


def rolling_beta(stock_ret, bench_ret, window=60):
    """滚动 Beta：个股收益对基准收益回归的斜率（显式滚动协方差，按行对齐）。"""
    b = bench_ret.reindex(stock_ret.index)
    mean_r = stock_ret.rolling(window).mean()
    mean_b = b.rolling(window).mean()
    var_b = b.rolling(window).var()
    cov = (stock_ret.mul(b, axis=0)).rolling(window).mean() - \
        mean_r.mul(mean_b, axis=0)
    return cov.div(var_b, axis=0)


def build_barra(close, volume, mkt, pe, pb, bench_ret, window=60):
    ret = close.pct_change()
    f = {}

    # 1. Size：对数市值
    size = np.log(mkt)
    f["Size"] = size
    # 5. Non-linear Size：市值的立方（截面去均值后再立方，捕捉非线性）
    f["Non-linear Size"] = (size - size.mean(axis=1)) ** 3

    # 2. Beta
    f["Beta"] = rolling_beta(ret, bench_ret, window)

    # 3. Momentum：长期动量，剔除最近20日
    f["Momentum"] = close.shift(20) / close.shift(120) - 1

    # 4. Residual Volatility：Beta 回归残差波动 + 已实现波动
    predicted = f["Beta"].mul(bench_ret.reindex(f["Beta"].index), axis=0)
    resid_ret = ret - predicted
    f["Residual Volatility"] = resid_ret.rolling(window).std()
    f["Realized Volatility"] = ret.rolling(window).std()

    # 6. Liquidity：成交额（负向，低流动性）+ 换手代理
    dollar_vol = (close * volume)
    f["Liquidity"] = np.log(dollar_vol.rolling(20).mean().replace(0, np.nan))
    f["Turnover"] = volume.rolling(20).mean() / mkt

    # 7. BTP / 8. Earnings
    f["BTP"] = 1.0 / pb
    f["EP"] = (1.0 / pe).clip(lower=0)

    # 9. Leverage：无负债数据，给占位（真实需财报负债/资产）
    f["Leverage"] = close * np.nan

    # 10. Growth：盈利收益率改善（成长代理）
    f["Growth"] = f["EP"].diff(60)

    return f
