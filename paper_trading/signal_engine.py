# -*- coding: utf-8 -*-
"""
核心信号引擎（回测/模拟共用）：
严格复刻 capstone/capstone_strategy.ipynb 的数据、因子、标签、walk-forward
LightGBM 与 Top30 等权组合逻辑，保证同一份代码既可产出回测基准，
又能在任意调仓日以 Point-in-Time（PIT）方式独立生成目标权重。

口径（与 capstone 完全一致，勿随意改动）：
  数据 : qlib 日频 hs300，start 2018-01-01，不写 end_time
  因子 : REV5 / VOL20 / SIZE / MOM60
  标签 : 未来 5 日收益的截面 rank
  模型 : LightGBM，按年 walk-forward（测试年用该年之前所有年训练，年内复用）
  组合 : 每月最后一个交易日取预测分 Top30 等权，调仓日整行清零再赋权
"""
import sys
import numpy as np
import pandas as pd

# 确保可导入项目根的 config / l2.config
_ROOT = r"C:\Users\gucao\Desktop\Qlib"
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import qlib  # noqa: E402
import config  # noqa: E402
from l2.config import SEED, load_universe  # noqa: E402

# 回测交易成本：单边 0.001，双边按换手计 0.002（与 capstone 一致）
COST_RATE = 0.002
TOP_N = 30
DATA_START = "2018-01-01"


def _cross_scale(df):
    """截面标准化：减当日截面均值、除当日截面标准差。"""
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)


class SignalEngine:
    """加载一次数据，按需训练年度模型并缓存，产出回测基准与 PIT 信号。"""

    def __init__(self, universe="hs300"):
        qlib.init(provider_uri=str(config.QLIB_DAILY_DIR),
                  joblib_backend="threading")
        from qlib.data import D
        self.D = D
        self.codes = load_universe(universe)
        self._load_panel()
        self._build_factors_and_xy()
        self._models = {}          # test_year -> 已训练模型
        self._pred_wide = None
        self._weights = None

    # ---------- 数据 ----------
    def _load_wide(self, field):
        panel = self.D.features(self.codes, [field],
                                start_time=DATA_START, freq="day")
        return panel[field].unstack("instrument")

    def _load_panel(self):
        self.close = self._load_wide("$close")
        self.volume = self._load_wide("$volume")
        self.mktcap = self._load_wide("$mkt_cap_ard")
        self.ret = self.close.pct_change(fill_method=None)

    def _build_factors_and_xy(self):
        self.factors = {
            "REV5": self.close.pct_change(5, fill_method=None),
            "VOL20": self.ret.rolling(20).std(),
            "SIZE": np.log(self.mktcap),
            "MOM60": self.close.pct_change(60, fill_method=None),
        }
        fwd5 = self.ret.shift(-5)
        label = fwd5.rank(axis=1)
        # 完整特征：实盘/模拟在任意日只需要因子即可预测，不依赖当日标签
        X_full = pd.concat([_cross_scale(v).stack()
                            for v in self.factors.values()], axis=1)
        X_full.columns = list(self.factors.keys())
        y = label.stack()
        self.X_full = X_full
        # 训练样本：特征与标签都存在的日期（末尾无未来收益的日期不参与训练）
        common = X_full.index.intersection(y.index)
        self.X, self.y = X_full.loc[common], y.loc[common]
        self._dates = pd.DatetimeIndex(self.X.index.get_level_values(0))
        self.years = sorted(set(self._dates.year))
        self.test_years = self.years[2:]      # 与 capstone 相同，跳过前两年

    # ---------- 模型 ----------
    def _train_model(self, test_year):
        """训练并缓存 test_year 所用模型：仅用 year < test_year 的样本。"""
        if test_year in self._models:
            return self._models[test_year]
        import lightgbm as lgb
        tr = self._dates.year < test_year
        model = lgb.LGBMRegressor(n_estimators=200, max_depth=4,
                                  learning_rate=0.05, subsample=0.8,
                                  verbose=-1, random_state=SEED)
        model.fit(self.X[tr], self.y[tr])
        self._models[test_year] = model
        return model

    # ---------- 回测基准（完整复刻 capstone） ----------
    def predict_wide(self):
        """全样本 walk-forward 预测宽表（capstone 的 pred_wide）。"""
        if self._pred_wide is not None:
            return self._pred_wide
        full_dates = pd.DatetimeIndex(
            self.X_full.index.get_level_values(0))
        preds = []
        for ty in self.test_years:
            model = self._train_model(ty)
            te = full_dates.year == ty
            p = pd.Series(model.predict(self.X_full[te]),
                          index=self.X_full.index[te])
            preds.append(p)
        pred = pd.concat(preds).sort_index()
        self._pred_wide = pred.unstack("instrument")
        return self._pred_wide

    def month_end_dates(self, start=None):
        """每月最后一个交易日（对齐 capstone resample('ME').last）。"""
        idx = self.close.index
        s = pd.Series(idx, index=idx).resample("ME").last().dropna()
        dates = pd.DatetimeIndex(s.values)
        if start is not None:
            dates = dates[dates >= pd.Timestamp(start)]
        return dates

    def backtest_weights(self):
        """完整回测权重宽表（capstone 的 weights）。"""
        if self._weights is not None:
            return self._weights
        pred_wide = self.predict_wide()
        month_end = self.month_end_dates()
        target = pd.DataFrame(np.nan, index=self.close.index,
                              columns=self.close.columns, dtype=float)
        for d in month_end:
            if d not in pred_wide.index:
                continue
            row = pred_wide.loc[d].dropna()
            top = row.nlargest(TOP_N).index
            target.loc[d, :] = 0.0
            target.loc[d, top] = 1 / len(top)
        self._weights = target.ffill().fillna(0)
        return self._weights

    def backtest_returns(self):
        """
        回测日收益序列（capstone 口径），返回 dict：
          gross/net/bench 日收益、cost 日成本、nav/bench_nav 净值。
        """
        weights = self.backtest_weights()
        w_prev = weights.shift(1).fillna(0)
        turnover = (weights - w_prev).abs().sum(axis=1)
        cost = turnover * COST_RATE
        gross = (w_prev * self.ret).sum(axis=1)
        net = gross - cost
        bench = self.ret.mean(axis=1).fillna(0)
        return {
            "gross": gross, "cost": cost, "net": net, "bench": bench,
            "nav": (1 + net.fillna(0)).cumprod(),
            "bench_nav": (1 + bench).cumprod(),
            "turnover": turnover,
        }

    def backtest_returns_bh(self, init_cash=1_000_000.0):
        """
        Buy-and-hold 口径回测（与 PaperBroker 同源的份额记账，独立实现）：
        初始全现金，调仓日按收盘价把目标权重换算为份额并扣双边成本；
        非调仓日份额固定、权重随价格漂移。这是与真实/模拟持仓一致的净值口径。
        返回 dict：net_bh 日收益、nav_bh 归一净值。
        """
        weights = self.backtest_weights()
        rebal_days = set(self.month_end_dates())
        cash = float(init_cash)
        shares = pd.Series(0.0, index=self.close.columns, dtype=float)
        rets, navs = [], []
        prev_value = None
        for d in self.close.index:
            prices = self.close.loc[d]
            if d in rebal_days and d in weights.index and weights.loc[d].abs().sum() > 0:
                v_before = cash + float((shares * prices).sum())
                tw = weights.loc[d]
                # 预留成本后再分配，与 PaperBroker 同口径
                target0 = (tw * v_before / prices).replace(
                    [np.inf, -np.inf], np.nan).fillna(0.0)
                cost0 = float(((target0 - shares).abs() * prices).sum()) \
                    * COST_RATE
                target_shares = (tw * (v_before - cost0) / prices).replace(
                    [np.inf, -np.inf], np.nan).fillna(0.0)
                dsh = target_shares - shares
                cost = float((dsh.abs() * prices).sum()) * COST_RATE
                cash += float((-dsh * prices).sum()) - cost
                shares = target_shares
            value = cash + float((shares * prices).sum())
            rets.append(value / prev_value - 1
                        if prev_value is not None else np.nan)
            navs.append(value)
            prev_value = value
        net_bh = pd.Series(rets, index=self.close.index)
        nav_raw = pd.Series(navs, index=self.close.index)
        base = nav_raw.dropna().iloc[0]
        nav_bh = nav_raw / base
        return {"net_bh": net_bh, "nav_bh": nav_bh}

    # ---------- PIT 信号（模拟盘独立路径） ----------
    def target_weights_pit(self, asof):
        """
        以 PIT 方式独立生成 asof 调仓日的目标权重：
        只用 asof 当年之前数据训练的模型，对 asof 截面预测，取 Top30 等权。
        返回权重 Series（index 为全部股票，落选为 0）。
        """
        asof = pd.Timestamp(asof)
        if asof.year not in self.test_years:
            raise ValueError("日期 %s 尚未进入 walk-forward 预测范围" % asof.date())
        model = self._train_model(asof.year)
        xrow = self.X_full.xs(asof, level=0)
        p = pd.Series(model.predict(xrow), index=xrow.index).dropna()
        top = p.nlargest(TOP_N).index
        w = pd.Series(0.0, index=self.close.columns, dtype=float)
        w.loc[top] = 1.0 / len(top)
        return w

    def is_rebalance_day(self, day):
        """day 是否为当月最后一个交易日。"""
        day = pd.Timestamp(day)
        return day in set(self.month_end_dates())


# 模块级单例：整个进程只初始化一次 qlib、只加载一次数据
_ENGINE = None


def get_engine(universe="hs300"):
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = SignalEngine(universe=universe)
    return _ENGINE
