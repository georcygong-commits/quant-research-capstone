# -*- coding: utf-8 -*-
"""线性模型：OLS / Ridge / Lasso，适配选股（支持截面 rank 标签）。"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso


def rank_label(d, label="label"):
    """把标签在每个截面上转为百分位（学排序，与 IC 目标一致）。"""
    d = d.copy()
    d["y_rank"] = d.groupby(level="datetime")[label].rank(pct=True)
    return d


def fit_linear(tr, feat, kind="ridge", alpha=1.0, use_rank_label=True):
    """训练线性模型。kind: ols / ridge / lasso。"""
    ycol = "y_rank" if use_rank_label else "label"
    if use_rank_label:
        tr = rank_label(tr)
    if kind == "ols":
        m = LinearRegression()
    elif kind == "ridge":
        m = Ridge(alpha=alpha)
    elif kind == "lasso":
        m = Lasso(alpha=alpha, max_iter=3000)
    else:
        raise ValueError(kind)
    m.fit(tr[feat], tr[ycol])
    return m


def coef_table(m, feat):
    return pd.Series(m.coef_, index=feat).sort_values(ascending=False)
