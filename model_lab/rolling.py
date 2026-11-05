# -*- coding: utf-8 -*-
"""滚动训练（Walk-Forward）：定期重训，产出连续样本外预测。"""
import numpy as np
import pandas as pd
from model_lab.linear import rank_label
from model_lab.tree import fit_lgb
from model_lab.linear import fit_linear


def make_repoints(df, freq_months=6, start_year=2023):
    """生成每 freq_months 个月一个重训点（月末）。"""
    last = df.index.get_level_values("datetime").max()
    pts = []
    y, m = start_year, 6
    while True:
        ts = pd.Timestamp(year=y, month=m, day=28)
        if ts >= last:
            break
        pts.append(ts)
        m += freq_months
        while m > 12:
            m -= 12; y += 1
    return pts


def walk_forward(df, feat, model="lgb", repoints=None, seed=42):
    """滚动训练，返回样本外预测 Series。"""
    df_r = rank_label(df)
    if repoints is None:
        repoints = make_repoints(df)
    last = df.index.get_level_values("datetime").max()
    all_pred = []
    for i, rp in enumerate(repoints):
        nxt = repoints[i + 1] if i + 1 < len(repoints) else last
        dts = df.index.get_level_values("datetime")
        tr = df_r[dts <= rp]
        te = df[(dts > rp) & (dts <= nxt)]
        if len(tr) == 0 or len(te) == 0:
            continue
        lvl = tr.index.get_level_values("datetime")
        va = tr[lvl > rp - pd.Timedelta(days=180)]
        if model == "lgb":
            m = fit_lgb(tr, va, feat, "y_rank", seed=seed)
            pred = m.predict(te[feat])
        else:
            m = fit_linear(tr, feat, "ridge", 1.0, use_rank_label=True)
            pred = m.predict(te[feat])
        all_pred.append(pd.Series(pred, index=te.index))
    return pd.concat(all_pred).sort_index()
