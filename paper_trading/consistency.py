# -*- coding: utf-8 -*-
"""
回测/实盘一致性检验：
读取 tracker 产出的模拟盘记录，与同一引擎的回测基准在重叠区间逐项对账：
  1) 调仓目标权重：PIT 独立信号 vs 回测权重（应完全一致，全区间严格）
  2) 稳态日收益/净值：首个常规调仓日之后，同口径 buy-and-hold 应仅浮点级误差
启动边界（建仓日至首个常规调仓日）因模拟盘从现金建仓、回测从旧仓再平衡，
总资产基数不同，差异单独量化、不参与严格判定，并在首个常规调仓日收敛。
"""
import numpy as np
import pandas as pd

from .signal_engine import get_engine
from .tracker import _STATE_DIR

# 对账阈值
TOL_WEIGHT = 1e-12      # 目标权重理论上应完全相同
TOL_DAILY = 1e-6        # 稳态日收益绝对误差
TOL_NAV = 1e-6          # 稳态归一净值相对偏离


def _perf(returns):
    r = returns.dropna()
    if len(r) == 0:
        return {"年化": np.nan, "夏普": np.nan, "最大回撤": np.nan,
                "累计收益": np.nan}
    ann = (1 + r).prod() ** (252 / len(r)) - 1
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
    cum = (1 + r).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    return {"年化": ann, "夏普": sharpe, "最大回撤": mdd,
            "累计收益": cum.iloc[-1] - 1}


def run_consistency(state_dir=_STATE_DIR, engine=None):
    eng = engine or get_engine()
    nav = pd.read_csv(state_dir / "nav.csv", parse_dates=["date"])
    trades = pd.read_csv(state_dir / "trades.csv", parse_dates=["date"])
    bt_weights = eng.backtest_weights()
    bh = eng.backtest_returns_bh()

    # 划分启动边界与稳态：首个常规调仓日（建仓日之后的第一个调仓日）
    reb_dates = nav.loc[nav["turnover"] > 0, "date"].reset_index(drop=True)
    build_day = reb_dates.iloc[0]
    steady_start = reb_dates.iloc[1]
    is_steady = nav["date"] > steady_start

    report = {"建仓日": build_day.date().isoformat(),
              "稳态起点": steady_start.date().isoformat()}

    # 1) 调仓目标权重一致性（全区间严格）
    weight_max_diff = 0.0
    weight_mismatch_days = 0
    for d, grp in trades.groupby("date"):
        pit = pd.Series(grp["target_weight"].values,
                        index=grp["code"]).astype(float)
        if d not in bt_weights.index:
            continue
        ref = bt_weights.loc[d]
        idx = pit.index.union(ref.index)
        diff = (pit.reindex(idx).fillna(0)
                - ref.reindex(idx).fillna(0)).abs().max()
        weight_max_diff = max(weight_max_diff, float(diff))
        if diff > TOL_WEIGHT:
            weight_mismatch_days += 1
    report["权重最大绝对差异"] = weight_max_diff
    report["权重不一致调仓日数"] = weight_mismatch_days

    # 同口径 BH 日收益
    bh_ret = bh["net_bh"].reindex(nav["date"]).values
    bh_nav = bh["nav_bh"].reindex(nav["date"]).reset_index(drop=True)
    paper_nav = nav["paper_nav"].reset_index(drop=True)
    daily = np.abs(nav["paper_ret"].values - bh_ret)

    # 全区间归一（启动边界口径，基准为建仓日）
    norm_p_all = paper_nav / paper_nav.iloc[0]
    norm_b_all = bh_nav / bh_nav.iloc[0]
    navrel_all = np.abs((norm_p_all - norm_b_all) / norm_b_all).values

    # 稳态起点重新归一（消除启动日基数差异，反映稳态后真实一致性）
    steady_mask = is_steady.values
    first_steady = int(np.argmax(steady_mask))
    norm_p_s = paper_nav / paper_nav.iloc[first_steady]
    norm_b_s = bh_nav / bh_nav.iloc[first_steady]
    navrel_s = np.abs((norm_p_s - norm_b_s) / norm_b_s).values

    # 2) 启动边界（单独报告，不参与判定）
    boundary = ~steady_mask
    report["启动边界_日收益最大差异"] = float(np.nanmax(daily[boundary]))
    report["启动边界_净值最大偏离"] = float(np.nanmax(navrel_all[boundary]))

    # 3) 稳态对账（严格判定）
    report["稳态_日收益最大差异"] = float(np.nanmax(daily[steady_mask]))
    report["稳态_净值最大偏离"] = float(np.nanmax(navrel_s[steady_mask]))

    # 4) 固定权重口径对照（已知口径差异，不参与判定）
    fw_daily = (nav["paper_ret"] - nav["backtest_ret"]).abs()
    report["对照_固定权重日收益最大差异"] = float(fw_daily.max())

    # 5) 判定：权重一致 + 稳态日收益/净值一致
    checks = {
        "目标权重一致": weight_max_diff <= TOL_WEIGHT,
        "稳态日收益一致": report["稳态_日收益最大差异"] <= TOL_DAILY,
        "稳态净值一致": report["稳态_净值最大偏离"] <= TOL_NAV,
    }
    report["对账结论"] = "PASS" if all(checks.values()) else "FAIL"
    report["逐项判定"] = checks

    # 6) 模拟盘 vs 基准绩效（跟踪区间）
    summary = pd.DataFrame({
        "模拟盘": _perf(nav["paper_ret"]),
        "BH回测": _perf(pd.Series(bh_ret, index=nav["date"])),
        "等权基准": _perf(nav["bench_ret"]),
    }).round(4)

    # 保存对账结果
    flat = {k: v for k, v in report.items() if k != "逐项判定"}
    pd.Series(flat).to_csv(state_dir / "consistency.csv",
                           header=["value"], encoding="utf-8-sig")
    return report, summary
