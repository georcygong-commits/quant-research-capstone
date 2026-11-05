# -*- coding: utf-8 -*-
"""强化学习与生成式（自实现）：tabular Q-learning 交易决策。
生成式因子（AlphaGen 用 RL 生成表达式）的思路在 notebook 讲解。"""
import numpy as np


class QLearningAgent:
    """单步 Q-learning：离散状态、离散动作，Q 表存储。"""

    def __init__(self, n_actions=2, lr=0.1, gamma=0.95, eps=0.2, seed=42):
        self.n_actions = n_actions
        self.lr, self.gamma, self.eps = lr, gamma, eps
        self.rng = np.random.default_rng(seed)
        self.q = {}

    def _row(self, s):
        if s not in self.q:
            self.q[s] = np.zeros(self.n_actions)
        return self.q[s]

    def choose(self, s, greedy=False):
        if (not greedy) and self.rng.random() < self.eps:
            return int(self.rng.integers(self.n_actions))
        row = self._row(s)
        return int(np.argmax(row))

    def update(self, s, a, r, s_next, done=False):
        row = self._row(s)
        target = r if done else r + self.gamma * np.max(self._row(s_next))
        row[a] += self.lr * (target - row[a])


def discretize(series, n_bins=3):
    """把连续序列按分位数离散化为整数状态。"""
    qs = np.unique(np.quantile(np.asarray(series, float),
                               np.linspace(0, 1, n_bins + 1)))
    return np.digitize(np.asarray(series, float), qs[1:-1])


def train_qlearning(agent, state_seq, fwd_rets):
    """在单资产上训练：动作 1=持有获得下期收益，0=空仓。

    契约（防前视）：state_seq[t] 是 t 时刻收盘已知的状态，fwd_rets[t] 是
    t 到 t+1 的下一期收益；两者等长、逐位对齐，决策只依赖 t 及之前信息。

    返回（策略累计收益、买入持有收益、动作序列）。
    """
    state_seq = np.asarray(state_seq)
    fwd_rets = np.asarray(fwd_rets, float)
    actions = []
    strat = []
    for t in range(len(state_seq) - 1):
        s = int(state_seq[t])
        a = agent.choose(s)
        r = a * fwd_rets[t]
        agent.update(s, a, r, int(state_seq[t + 1]))
        actions.append(a)
        strat.append(r)
    strat = np.asarray(strat)
    bh = fwd_rets[:len(strat)]
    return (np.prod(1 + strat) - 1, np.prod(1 + bh) - 1, np.asarray(actions))
