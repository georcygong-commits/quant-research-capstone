# -*- coding: utf-8 -*-
"""简单组合权重规则：等权、Top-N、打分加权、多空。"""
import numpy as np
import pandas as pd


def equal_weight(scores):
    """对所有有分数的股票等权（满仓多头）。"""
    s = scores.dropna()
    return pd.Series(1.0 / len(s), index=s.index)


def topn_equal(scores, n=30):
    """分数最高的 n 只等权。"""
    top = scores.dropna().nlargest(n)
    return pd.Series(1.0 / len(top), index=top.index)


def score_weight(scores):
    """分数非负化后按分数加权（越看好权重越大）。"""
    s = scores.dropna()
    s = s - s.min()
    if s.sum() > 0:
        return s / s.sum()
    return equal_weight(scores)


def long_short_weights(scores, n=30):
    """多空：top n 等权多头、bottom n 等权空头（净敞口 0）。"""
    s = scores.dropna()
    longs = s.nlargest(n).index
    shorts = s.nsmallest(n).index
    w = pd.Series(0.0, index=s.index)
    w[longs] = 1.0 / n
    w[shorts] = -1.0 / n
    return w


def portfolio_return(weights, returns):
    """给定权重和截面收益，算组合收益。"""
    common = weights.index.intersection(returns.dropna().index)
    return (weights[common] * returns[common]).sum()
