# -*- coding: utf-8 -*-
"""
教学用遗传规划因子挖掘（不依赖 gplearn）：
- 表达式树：用嵌套 tuple 表示，叶子 ("base", 名称)，内部 (op, 子表达式...)
- 算子：截面/时序 rank、均值、标准差、延迟、加减乘、负号、对数等
- 随机生成、交叉、变异；适应度用 IC 均值。
"""
import copy
import numpy as np
import pandas as pd

# ---------------- 算子 ----------------
def _cs_rank(x):
    return x.rank(axis=1, pct=True) * 2 - 1

def _neg(x):
    return -x

def _abs(x):
    return x.abs()

def _delay1(x):
    return x.shift(1)

def _delay5(x):
    return x.shift(5)

def _ts_mean5(x):
    return x.rolling(5).mean()

def _ts_mean10(x):
    return x.rolling(10).mean()

def _ts_std10(x):
    return x.rolling(10).std()

def _ts_max10(x):
    return x.rolling(10).max()

def _ts_min10(x):
    return x.rolling(10).min()

# 单目算子（不含常数参数，便于遗传）
UNARY = {
    "cs_rank": (_cs_rank, 1),
    "neg": (_neg, 1),
    "abs": (_abs, 1),
    "delay1": (_delay1, 1),
    "delay5": (_delay5, 1),
    "ts_mean5": (_ts_mean5, 1),
    "ts_mean10": (_ts_mean10, 1),
    "ts_std10": (_ts_std10, 1),
    "ts_max10": (_ts_max10, 1),
    "ts_min10": (_ts_min10, 1),
}

# 双目算子
BINARY = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
}


def make_base(close, volume, mkt, pe, pb):
    """构造叶子节点可用的基础因子面板。"""
    ret1 = close.pct_change()
    return {
        "ret5": close.pct_change(5),
        "ret20": close.pct_change(20),
        "vol10": ret1.rolling(10).std(),
        "logmkt": np.log(mkt),
        "turn10": np.log((volume.rolling(10).mean() / mkt).replace(0, np.nan)),
        "bp": 1.0 / pb,
    }


# ---------------- 表达式求值 ----------------
def evaluate(expr, base):
    """递归求值表达式树，返回宽表。"""
    if expr[0] == "base":
        return base[expr[1]]
    op = expr[0]
    if op in UNARY:
        return UNARY[op][0](evaluate(expr[1], base))
    if op in BINARY:
        return BINARY[op](evaluate(expr[1], base), evaluate(expr[2], base))
    raise ValueError(op)


def expr_to_str(expr):
    if expr[0] == "base":
        return expr[1]
    if expr[0] in UNARY:
        return "%s(%s)" % (expr[0], expr_to_str(expr[1]))
    return "%s(%s, %s)" % (expr[0], expr_to_str(expr[1]), expr_to_str(expr[2]))


# ---------------- 随机生成 / 交叉 / 变异 ----------------
BASE_NAMES = ["ret5", "ret20", "vol10", "logmkt", "turn10", "bp"]

def gen_expr(depth=0, max_depth=3, rng=None):
    rng = rng or np.random.default_rng()
    if depth >= max_depth or (depth > 0 and rng.random() < 0.4):
        return ("base", rng.choice(BASE_NAMES))
    if rng.random() < 0.7 or depth >= max_depth - 1:
        op = rng.choice(list(UNARY.keys()))
        return (op, gen_expr(depth + 1, max_depth, rng))
    op = rng.choice(list(BINARY.keys()))
    return (op, gen_expr(depth + 1, max_depth, rng),
            gen_expr(depth + 1, max_depth, rng))


def _all_nodes(expr):
    nodes = [expr]
    if expr[0] != "base":
        for sub in expr[1:]:
            nodes.extend(_all_nodes(sub))
    return nodes


def crossover(e1, e2, rng):
    a = copy.deepcopy(e1)
    # 用 e2 的一个随机子树替换 a 中的随机节点
    donor = _all_nodes(copy.deepcopy(e2))[rng.integers(len(_all_nodes(e2)))]
    return _replace_random(a, donor, rng)


def _replace_random(expr, donor, rng):
    if expr[0] == "base":
        return copy.deepcopy(donor)
    if rng.random() < 0.3:
        return copy.deepcopy(donor)
    new = [expr[0]]
    for sub in expr[1:]:
        new.append(_replace_random(sub, donor, rng))
    return tuple(new)


def mutate(expr, rng, max_depth=3):
    return _replace_random(expr, gen_expr(0, max_depth, rng), rng)


# ---------------- 遗传规划主循环 ----------------
def _fitness(expr, base, fwd, sample):
    v = _cs_rank(evaluate(expr, base))
    common = v.index.intersection(fwd.index).intersection(sample)
    ics = []
    for dt in common:
        a, b = v.loc[dt], fwd.loc[dt]
        ok = a.notna() & b.notna()
        if ok.sum() > 10:
            ics.append(a[ok].corr(b[ok], method="spearman"))
    if not ics:
        return 0.0, 0.0
    m = np.mean(ics)
    if np.isnan(m):
        return 0.0, 0.0
    return abs(m), m


def run_gp(base, fwd, n_pop=40, n_gen=6, n_elite=5, sample_step=5,
           seed=0, max_depth=3, verbose=False):
    """
    遗传规划挖掘因子。
    fwd: 未来收益宽表（建议用 5 日）；sample_step: 采样间隔（与持有期一致，避免重叠）。
    返回 (最优结果列表, 每代最优适应度)。
    """
    rng = np.random.default_rng(seed)
    dates = fwd.dropna(how="all").index
    sample = dates[::sample_step]
    pop = [gen_expr(0, max_depth, rng) for _ in range(n_pop)]
    history = []
    scored = []
    for g in range(n_gen):
        scored = []
        for e in pop:
            try:
                a, m = _fitness(e, base, fwd, sample)
                scored.append((a, m, e))
            except Exception:
                scored.append((0.0, 0.0, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        history.append(scored[0][0])
        new = [scored[i][2] for i in range(n_elite)]
        pool_size = min(n_elite * 3, len(scored))
        while len(new) < n_pop:
            if rng.random() < 0.7:
                p1 = scored[rng.integers(pool_size)][2]
                p2 = scored[rng.integers(pool_size)][2]
                new.append(crossover(p1, p2, rng))
            else:
                new.append(mutate(scored[rng.integers(pool_size)][2], rng, max_depth))
        pop = new
        if verbose:
            print("第%d代 最优|IC|=%.4f" % (g, history[-1]))
    return scored, history
