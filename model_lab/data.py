# -*- coding: utf-8 -*-
"""模型预测的数据构建：因子 -> 截面标准化长表 -> 标签 -> 滚动划分。"""
import numpy as np
import pandas as pd
from factor_lab.library import build_factors


def _cs_rank(f):
    """截面 rank 标准化到 [-1,1]（宽表）。"""
    return f.rank(axis=1, pct=True) * 2 - 1


def build_panel(close, volume, mkt, pe, pb, label_horizon=1,
                factor_names=None):
    """
    构造建模用长表：index=(datetime, instrument)，列=标准化因子 + label。
    label 为未来 label_horizon 日收益（小数）。
    """
    facs = build_factors(close, volume, mkt, pe, pb)
    if factor_names is not None:
        facs = {k: v for k, v in facs.items() if k in factor_names}
    data = {}
    for name, f in facs.items():
        data[name] = _cs_rank(f).stack()
    label = (close.shift(-label_horizon) / close - 1).stack()
    df = pd.DataFrame(data)
    df = df.fillna(0.0)  # 特征缺失（数据不足/停牌）填中性值 0
    df["label"] = label
    df.index = df.index.set_names(["datetime", "instrument"])
    df = df.dropna(subset=["label"])
    return df


def split_by_date(df, train_end, valid_end, test_end=None):
    """按日期切分（训练 <= train_end < 验证 <= valid_end < 测试）。"""
    dates = df.index.get_level_values("datetime")
    tr = df[dates <= train_end]
    va = df[(dates > train_end) & (dates <= valid_end)]
    te = df[dates > valid_end] if test_end is None else df[
        (dates > valid_end) & (dates <= test_end)]
    return tr, va, te


def predict_ic(pred, label):
    """按日期计算预测值的截面 RankIC。"""
    g = pd.DataFrame({"pred": pred, "label": label})
    g.index = label.index
    out = []
    for dt, sub in g.groupby(level="datetime"):
        if len(sub) > 10:
            out.append(sub["pred"].corr(sub["label"], method="spearman"))
    return pd.Series(out)
