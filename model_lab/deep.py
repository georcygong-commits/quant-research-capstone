# -*- coding: utf-8 -*-
"""深度学习模型：前馈网络（MLP）可用 sklearn 演示；LSTM/Transformer 给出 torch 框架。"""
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor


def fit_mlp(tr, feat, label="y_rank", hidden=(64, 32),
            alpha=0.01, max_iter=80, seed=42):
    """前馈神经网络（即 Qlib 的 DNN 模型思想）。"""
    m = MLPRegressor(hidden_layer_sizes=hidden, activation="relu",
                     alpha=alpha, learning_rate_init=0.001,
                     max_iter=max_iter, early_stopping=True,
                     random_state=seed)
    m.fit(tr[feat], tr[label])
    return m


# ---------- 以下为 PyTorch 序列模型框架（需 pip install torch）----------
TORCH_LSTM = '''
import torch, torch.nn as nn

class LSTMModel(nn.Module):
    """输入 [批次, 时间步, 特征数] 的量价序列，输出每只股票的预测。"""
    def __init__(self, n_feat, hidden=64, n_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, n_layers,
                            batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)

class ALSTMModel(LSTMModel):
    """在 LSTM 输出上加注意力（Qlib 常用 ALSTM）。"""
    def __init__(self, n_feat, hidden=64, n_layers=2):
        super().__init__(n_feat, hidden, n_layers)
        self.attn = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        w = torch.softmax(self.attn(out), dim=1)
        ctx = (w * out).sum(dim=1)
        return self.head(ctx).squeeze(-1)
'''
