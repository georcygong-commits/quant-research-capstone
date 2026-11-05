# -*- coding: utf-8 -*-
"""因子拥挤、衰减与容量。"""
import numpy as np
import pandas as pd


def rolling_ic_stats(ic, window=60):
    """滚动 IC 均值，用于观察因子衰减与失效。"""
    return pd.DataFrame({"滚动IC": ic.rolling(window).mean(),
                         "滚动IR": ic.rolling(window).mean() /
                         ic.rolling(window).std()})


def value_spread(factor, value_field, n_quantile=5):
    """
    估值价差：因子最高组与最低组的平均估值差（拥挤时价差往往收窄）。
    教学版：逐期取两端均值之差。
    """
    out = {}
    for dt in factor.index:
        f = factor.loc[dt].dropna()
        v = value_field.loc[dt].reindex(f.index)
        if len(f) < 20:
            continue
        hi = f.nlargest(max(5, len(f)//n_quantile)).index
        lo = f.nsmallest(max(5, len(f)//n_quantile)).index
        out[dt] = v.reindex(hi).mean() - v.reindex(lo).mean()
    return pd.Series(out)


def crowding_score(factor, value_field, window=120):
    """
    拥挤度（0~1 教学版）：综合因子暴露集中度（截面std）与近期表现，
    暴露越集中、近期涨得越多，越拥挤。
    """
    concentration = factor.std(axis=1)
    recent = factor.mean(axis=1).rolling(window).mean()
    zc = (concentration - concentration.rolling(window).mean()) / \
        concentration.rolling(window).std()
    zr = (recent - recent.rolling(window).mean()) / \
        recent.rolling(window).std()
    score = (zc.fillna(0) + zr.fillna(0)) / 2
    return (score - score.min()) / (score.max() - score.min())


def factor_capacity(avg_daily_adv, max_impact_bps=10, coef=1.0, daily_vol=0.02):
    """
    容量估算：给定可接受冲击 bps，用平方根模型反推参与率，
    再乘日均成交额得到可承载金额。
    """
    # impact = coef*vol*sqrt(part) -> part = (impact/(coef*vol))^2
    part = (max_impact_bps/10000 / (coef*daily_vol)) ** 2
    return avg_daily_adv * np.clip(part, 0, 1), part
