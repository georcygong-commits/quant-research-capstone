# -*- coding: utf-8 -*-
"""
因子预处理：去极值（3σ/MAD/分位）、标准化（Z-score/Rank）、缺失处理。
所有函数既可作用于一维截面，也可通过 process_factor 作用于整个因子面板。
"""
import numpy as np
import pandas as pd


# ---------- 去极值（一维） ----------
def winsorize_3sigma(x, n=3):
    x = pd.Series(x).astype(float)
    mu, sd = x.mean(), x.std()
    return x.clip(mu - n * sd, mu + n * sd)


def winsorize_mad(x, n=3):
    x = pd.Series(x).astype(float)
    med = x.median()
    mad = (x - med).abs().median()
    bound = n * 1.4826 * mad
    return x.clip(med - bound, med + bound)


def winsorize_quantile(x, lower=0.01, upper=0.99):
    x = pd.Series(x).astype(float)
    return x.clip(x.quantile(lower), x.quantile(upper))


# ---------- 标准化（一维） ----------
def zscore(x):
    x = pd.Series(x).astype(float)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else x * 0


def rank_norm(x):
    x = pd.Series(x).astype(float)
    return x.rank(pct=True) * 2 - 1


# ---------- 面板处理 ----------
def process_factor(factor, winsor="mad", standardize="rank", fillna=0.0):
    """逐截面：去极值 -> 标准化 -> 缺失填中性值。"""
    def one_cross(x):
        x = x.astype(float)
        if winsor == "mad":
            x = winsorize_mad(x)
        elif winsor == "quantile":
            x = winsorize_quantile(x)
        elif winsor == "3sigma":
            x = winsorize_3sigma(x)
        if standardize == "rank":
            x = rank_norm(x)
        elif standardize == "zscore":
            x = zscore(x)
        return x.fillna(fillna)
    return factor.apply(one_cross, axis=1)
