# -*- coding: utf-8 -*-
"""
分层回测：截面分组、组内等权、定期调仓，输出各组收益、多空、净值、
年化、夏普、最大回撤、换手率。
"""
import numpy as np
import pandas as pd


def make_labels(factor, n_groups=5):
    """逐截面按分位数分组，缺失/常数截面不分组。"""
    def lab(x):
        x = x.astype(float)
        valid = x.notna()
        if valid.sum() < n_groups or x[valid].nunique() < 2:
            return pd.Series(np.nan, index=x.index)
        g = np.ceil(x.rank(pct=True) * n_groups).clip(1, n_groups)
        g[~valid] = np.nan
        return g
    return factor.apply(lab, axis=1)


def layered_returns(factor, daily_ret, n_groups=5):
    """返回各组每日等权收益（用 t-1 信号赚 t 收益）。"""
    hold = make_labels(factor, n_groups).shift(1)
    gr = pd.DataFrame({g: daily_ret.where(hold == g).mean(axis=1)
                       for g in range(1, n_groups + 1)})
    return gr.dropna(how="all"), hold


def perf_metrics(returns, periods=252):
    """计算年化、夏普、最大回撤。"""
    r = returns.dropna()
    ann = r.mean() * periods
    sharpe = r.mean() / r.std() * np.sqrt(periods) if r.std() > 0 else np.nan
    nav = (1 + r).cumprod()
    dd = (nav / nav.cummax() - 1).min()
    return pd.Series({"年化收益": ann, "夏普": sharpe, "最大回撤": dd})


def turnover(hold, n_groups=5):
    """各组单边换手率（成员变化比例的时间均值）。"""
    out = {}
    for g in range(1, n_groups + 1):
        m = (hold == g).astype(float)
        # 每日进入/退出该组的比例（相对组内数量）
        cnt = m.sum(axis=1).replace(0, np.nan)
        entered = (m.diff() > 0).sum(axis=1) / cnt
        out[g] = (entered.fillna(0)).mean()
    return pd.Series(out)
