# -*- coding: utf-8 -*-
"""
因子中性化：截面回归取残差，剔除行业 / 市值（或其他风格）暴露。
"""
import numpy as np
import pandas as pd


def cross_residual(y, X):
    """单截面回归返回残差。y: Series, X: DataFrame（不含截距）。"""
    valid = y.notna()
    if X is not None:
        valid = valid & X.notna().all(axis=1)
    yv = y[valid]
    Xv = X.loc[valid] if X is not None else pd.DataFrame(index=yv.index)
    Xmat = np.column_stack([np.ones(len(Xv)), Xv.values.astype(float)])
    beta, *_ = np.linalg.lstsq(Xmat, yv.values, rcond=None)
    resid = yv.values - Xmat @ beta
    return pd.Series(resid, index=yv.index)


def neutralize(factor, ind_map=None, style=None, by_industry=True,
               by_style=True):
    """
    逐截面中性化。
    factor: 因子宽表（date × code）。
    ind_map: {code: 行业}。
    style: 风格宽表（date × code），如 log 市值；或 DataFrame 多列风格的 dict。
    """
    if style is not None and not isinstance(style, dict):
        style = {"style": style}
    out = pd.DataFrame(index=factor.index, columns=factor.columns, dtype=float)
    for dt, y in factor.iterrows():
        cols = []
        if by_industry and ind_map is not None:
            ind = pd.Series([ind_map.get(c, "其他") for c in factor.columns],
                            index=factor.columns)
            cols.append(pd.get_dummies(ind, drop_first=True))
        if by_style and style:
            for name, sdf in style.items():
                cols.append(sdf.loc[dt].rename(name).to_frame())
        X = pd.concat(cols, axis=1) if cols else None
        r = cross_residual(y, X)
        out.loc[dt, r.index] = r.values
    return out
