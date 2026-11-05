# -*- coding: utf-8 -*-
"""绩效归因：Brinson（配置/选股/交互）与多因子归因。"""
import numpy as np
import pandas as pd


def brinson(weights_p, weights_b, returns, industry):
    """
    Brinson 归因（单期，按行业聚合）。
    weights_p/b: 组合/基准个股权重（Series）；
    returns: 个股收益（Series）；industry: 行业标签（Series）。
    返回每个行业的配置、选股、交互效应及合计。
    """
    df = pd.DataFrame({"wp": weights_p, "wb": weights_b, "r": returns,
                       "ind": industry}).dropna(subset=["ind", "r"])
    df["wp"] = df["wp"].fillna(0); df["wb"] = df["wb"].fillna(0)
    rows = []
    total_b = df["wb"].sum()
    for ind, g in df.groupby("ind"):
        wp = g["wp"].sum(); wb = g["wb"].sum()
        rp = (g["wp"] * g["r"]).sum() / wp if wp > 0 else 0
        rb = (g["wb"] * g["r"]).sum() / wb if wb > 0 else 0
        alloc = (wp - wb) * rb          # 配置效应（BHB）
        select = wb * (rp - rb)         # 选股效应
        interact = (wp - wb) * (rp - rb)  # 交互效应
        rows.append({"行业": ind, "配置": alloc, "选股": select,
                     "交互": interact, "合计": alloc + select + interact})
    out = pd.DataFrame(rows).set_index("行业")
    return out


def factor_attribution(weights_p, factor_returns, exposures):
    """
    多因子归因：组合在各因子上的暴露 × 因子收益。
    factor_returns: 当期因子收益（Series）；exposures: {因子: 暴露Series}。
    """
    names = list(factor_returns.index)
    port_exp = pd.Series({
        k: (weights_p.reindex(exposures[k].index).fillna(0)
            * exposures[k].fillna(0)).sum()
        for k in names})
    contrib = port_exp * factor_returns
    return pd.DataFrame({"组合暴露": port_exp, "因子收益": factor_returns,
                         "贡献": contrib})
