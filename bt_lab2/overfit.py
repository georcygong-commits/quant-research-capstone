# -*- coding: utf-8 -*-
"""
过拟合诊断与稳健性（M5.4）：
CSCV/PBO（回测过拟合概率，Bailey et al. 2014）、Deflated Sharpe Ratio
（Bailey & López de Prado 2014）、参数敏感性、置换检验与 Sharpe 的 bootstrap 区间。
"""
from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats


# ---------------- CSCV / PBO ----------------
def pbo_cscv(strategy_returns, n_blocks=6):
    """
    组合对称交叉验证（CSCV）：
      把时间轴分 n_blocks 块，枚举一半为训练、一半为验证，
      训练集选 Sharpe 最高策略，看其在验证集的相对排名；
      PBO = 该策略验证排名落到下半区（logit<0）的频率。
    strategy_returns: DataFrame(T, S)，每列为一个策略的收益序列。
    """
    R = np.asarray(strategy_returns, float)
    T, S = R.shape
    blocks = np.array_split(np.arange(T), n_blocks)
    half = n_blocks // 2
    logits = []
    for train_blocks in combinations(range(n_blocks), half):
        tr_idx = np.concatenate([blocks[b] for b in train_blocks])
        va_idx = np.concatenate([blocks[b] for b in range(n_blocks)
                                 if b not in train_blocks])
        tr_sr = R[tr_idx].mean(axis=0) / R[tr_idx].std(axis=0)
        s_star = int(np.argmax(tr_sr))
        va_sr = R[va_idx].mean(axis=0) / R[va_idx].std(axis=0)
        order = np.argsort(va_sr)
        rank = np.where(order == s_star)[0][0] + 1           # 1..S
        omega = rank / (S + 1)
        omega = np.clip(omega, 1e-6, 1 - 1e-6)
        logits.append(np.log(omega / (1 - omega)))
    logits = np.array(logits)
    return {"PBO": float(np.mean(logits < 0)),
            "logit_mean": float(logits.mean()),
            "n_combinations": len(logits), "logits": logits}


# ---------------- Deflated Sharpe ----------------
def deflated_sharpe(observed_sr, trial_srs, T, skew=0.0, excess_kurt=0.0):
    """
    Deflated Sharpe Ratio：在考虑「尝试了 N 个策略、存在选择偏差」后，
    观测 Sharpe 仍显著的概率（DSR，0~1）。
      SR0 = E[max of N trials]，用正态序贯统计量近似；
      DSR = Phi((SR_obs - SR0) sqrt(T-1) /
                sqrt(1 - skew SR + (excess_kurt/4) SR^2))
    """
    n_trials = len(trial_srs)
    sr_std = np.std(trial_srs, ddof=1) if n_trials > 1 else 0.0
    z = stats.norm.ppf(1 - 1.0 / n_trials)
    sr0 = z * sr_std
    denom = np.sqrt(1 - skew * observed_sr + (excess_kurt / 4.0) * observed_sr ** 2)
    dsr = stats.norm.cdf((observed_sr - sr0) * np.sqrt(T - 1) / denom)
    return {"DSR": float(dsr), "benchmark_SR0": float(sr0),
            "n_trials": int(n_trials)}


# ---------------- 参数敏感性 ----------------
def parameter_sensitivity(perf_grid):
    """
    perf_grid: 二维绩效矩阵（如不同参数组合的 Sharpe）。
    返回均值、标准差与变异系数，衡量绩效对参数的敏感程度。
    """
    a = np.asarray(perf_grid, float)
    return {"mean": float(a.mean()), "std": float(a.std()),
            "cv": float(a.std() / abs(a.mean())) if a.mean() else np.nan,
            "min": float(a.min()), "max": float(a.max())}


# ---------------- 置换检验 ----------------
def permutation_test(strategy_returns, signal, n_perm=200, seed=42):
    """
    打乱信号（破坏其与收益的对应关系），重算策略均值收益，
    原策略收益相对随机分布的分位即 p 值。
    """
    rng = np.random.default_rng(seed)
    r = np.asarray(strategy_returns, float)
    s = np.asarray(signal, float)
    base = np.mean(r * np.sign(s))
    perm = np.array([np.mean(r * np.sign(rng.permutation(s)))
                     for _ in range(n_perm)])
    p_value = float(np.mean(perm >= base))
    return {"observed": float(base), "p_value": p_value,
            "perm_mean": float(perm.mean())}


# ---------------- bootstrap Sharpe ----------------
def bootstrap_sharpe(returns, n_boot=1000, seed=42):
    """策略年化 Sharpe 的 bootstrap 置信区间（2.5%/97.5%）。"""
    rng = np.random.default_rng(seed)
    r = np.asarray(returns, float)
    T = len(r)
    srs = []
    for _ in range(n_boot):
        x = r[rng.integers(0, T, T)]
        srs.append(x.mean() / x.std() * np.sqrt(242))
    srs = np.array(srs)
    return {"sharpe": float(r.mean() / r.std() * np.sqrt(242)),
            "ci_low": float(np.quantile(srs, 0.025)),
            "ci_high": float(np.quantile(srs, 0.975))}
