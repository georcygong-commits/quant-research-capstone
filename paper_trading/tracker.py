# -*- coding: utf-8 -*-
"""
逐日驱动与状态持久化：
从启动日（默认 2024-01-31）逐日推进到最新交易日，月末调仓，
每日记录模拟净值、回测基准净值与实际持仓，增量写入 state/paper_trading，
支持断点续跑（重复运行只处理新增交易日）。
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

from .signal_engine import get_engine
from .paper_broker import PaperBroker

_STATE_DIR = Path(r"C:\Users\gucao\Desktop\Qlib\state\paper_trading")


class Tracker:
    def __init__(self, start_date="2024-01-31", init_cash=1_000_000.0,
                 state_dir=_STATE_DIR, engine=None):
        self.engine = engine or get_engine()
        self.start_date = pd.Timestamp(start_date)
        self.init_cash = float(init_cash)
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._bt = self.engine.backtest_returns()
        self._load_or_init()

    # ---------- 状态路径 ----------
    @property
    def meta_path(self):
        return self.state_dir / "meta.json"

    @property
    def nav_path(self):
        return self.state_dir / "nav.csv"

    @property
    def pos_path(self):
        return self.state_dir / "positions.csv"

    @property
    def trade_path(self):
        return self.state_dir / "trades.csv"

    def _load_or_init(self):
        eng = self.engine
        if self.meta_path.exists():
            meta = json.load(open(self.meta_path, encoding="utf-8"))
            self.broker = PaperBroker(self.init_cash, eng.close.columns)
            self.broker.cash = float(meta["cash"])
            if self.pos_path.exists():
                pos = pd.read_csv(self.pos_path, parse_dates=["date"])
                last = pos[pos["date"] == pos["date"].max()]
                sh = pd.Series(last["shares"].values,
                               index=last["code"]).astype(float)
                self.broker.shares = sh.reindex(eng.close.columns).fillna(0)
            self.last_done = pd.Timestamp(meta["last_date"])
        else:
            self.broker = PaperBroker(self.init_cash, eng.close.columns)
            self.last_done = None

    # ---------- 主循环 ----------
    def run(self, end_date=None):
        eng = self.engine
        idx = eng.close.index
        days = idx[idx >= self.start_date]
        if end_date is not None:
            days = days[days <= pd.Timestamp(end_date)]
        if self.last_done is not None:
            days = days[days > self.last_done]

        nav_rows, pos_rows, trade_rows = [], [], []
        prev_value = self.init_cash
        if self.last_done is not None and self.nav_path.exists():
            old = pd.read_csv(self.nav_path)
            prev_value = float(old["total_value"].iloc[-1])

        for d in days:
            prices = eng.close.loc[d]
            rebal_cost = 0.0
            turnover = 0.0
            if eng.is_rebalance_day(d):
                tw = eng.target_weights_pit(d)
                info = self.broker.rebalance(tw, prices)
                rebal_cost = info["cost_weight"]
                turnover = info["turnover_weight"]
                held = tw[tw > 0]
                for code, wgt in held.items():
                    trade_rows.append(
                        {"date": d, "code": code, "target_weight": float(wgt)})

            value = self.broker.total_value(prices)
            paper_ret = value / prev_value - 1.0
            w_actual = self.broker.current_weights(prices)
            held = w_actual[w_actual > 0]
            for code, wgt in held.items():
                pos_rows.append({
                    "date": d, "code": code,
                    "weight": float(wgt),
                    "shares": float(self.broker.shares.reindex(
                        [code]).iloc[0]),
                })

            nav_rows.append({
                "date": d,
                "paper_nav": self.broker.nav(prices),
                "total_value": value,
                "paper_ret": paper_ret,
                "backtest_nav": float(self._bt["nav"].get(d, np.nan)),
                "backtest_ret": float(self._bt["net"].get(d, np.nan)),
                "bench_nav": float(self._bt["bench_nav"].get(d, np.nan)),
                "bench_ret": float(self._bt["bench"].get(d, np.nan)),
                "turnover": turnover,
                "cost": rebal_cost,
            })
            prev_value = value
            self.last_done = d

        self._append(nav_rows, pos_rows, trade_rows)
        return {"processed": len(days), "last_date": self.last_done}

    # ---------- 持久化 ----------
    def _append(self, nav_rows, pos_rows, trade_rows):
        if nav_rows:
            nav_df = pd.DataFrame(nav_rows)
            nav_df.to_csv(self.nav_path, mode="a",
                          header=not self.nav_path.exists(), index=False)
        if pos_rows:
            pos_df = pd.DataFrame(pos_rows)
            pos_df.to_csv(self.pos_path, mode="a",
                          header=not self.pos_path.exists(), index=False)
        if trade_rows:
            tr_df = pd.DataFrame(trade_rows)
            tr_df.to_csv(self.trade_path, mode="a",
                         header=not self.trade_path.exists(), index=False)
        meta = {
            "start_date": self.start_date.date().isoformat(),
            "init_cash": self.init_cash,
            "cash": self.broker.cash,
            "last_date": self.last_done.date().isoformat()
            if self.last_done is not None else None,
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)


def run_tracker(start_date="2024-01-31", init_cash=1_000_000.0):
    """命令行入口：构建/恢复 tracker 并跑到最新交易日。"""
    t = Tracker(start_date=start_date, init_cash=init_cash)
    summary = t.run()
    return summary
