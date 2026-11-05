# -*- coding: utf-8 -*-
"""
多期 Brinson 与几何链接（M5.6）：
单期算术效应不能直接相加（复利），用 Cariño（1999）对数链接系数
把各期效应折算到对数收益空间，保证多期加总闭合。
"""
import numpy as np
import pandas as pd


def carino_coefficient(port_ret, bench_ret):
    """
    Cariño 链接系数：
      k_t = (R_t - B_t) / (ln(1+R_t) - ln(1+B_t))
    R_t=B_t 时取极限 (1+R_t)。
    """
    log_diff = np.log1p(port_ret) - np.log1p(bench_ret)
    if abs(log_diff) < 1e-12:
        return 1.0 + port_ret
    return (port_ret - bench_ret) / log_diff


def link_multi_period(period_effects, port_rets, bench_rets):
    """
    period_effects: DataFrame(T, 效应列[配置/选股/交互])，每期单期算术效应；
    port_rets/bench_rets: 每期组合/基准收益。
    返回链接后各效应（对数空间）、合计，并与实际对数超额核对闭合残差。
    """
    T = len(period_effects)
    k = np.array([carino_coefficient(port_rets[i], bench_rets[i]) for i in range(T)])
    linked = period_effects.div(k, axis=0)
    total_log_excess = np.log(np.prod(1 + np.asarray(port_rets))) - \
        np.log(np.prod(1 + np.asarray(bench_rets)))
    residual = float(linked.values.sum() - total_log_excess)
    summary = linked.sum()
    return {"linked": linked, "summary": summary,
            "carino_k": k, "log_excess": float(total_log_excess),
            "closure_residual": residual}


def geometric_excess_return(port_rets, bench_rets):
    """几何超额：组合累计相对基准累计的差额（可加对数形式）。"""
    return float(np.log(np.prod(1 + np.asarray(port_rets))) -
                 np.log(np.prod(1 + np.asarray(bench_rets))))
