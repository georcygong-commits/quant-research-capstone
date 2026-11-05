# -*- coding: utf-8 -*-
"""深度学习：torch 未装时用 sklearn MLP 跑真实网络，并附 numpy 教学版前馈网络。
LSTM/Transformer/GNN 的结构在 notebook 中讲解。"""
import numpy as np


def make_mlp(hidden=(32, 16), alpha=1e-3, seed=42):
    """sklearn 多层感知机（真实可训练的神经网络）。"""
    from sklearn.neural_network import MLPRegressor
    return MLPRegressor(hidden_layer_sizes=hidden, activation="relu",
                        alpha=alpha, max_iter=300, random_state=seed,
                        early_stopping=True)


class NumpyNet:
    """两层全连接网络（numpy 教学版）：线性→ReLU→线性，MSE + 梯度下降。

    用于理解前向传播与反向传播，不追求工程性能。
    """

    def __init__(self, n_input, n_hidden=16, lr=0.05, seed=42):
        rng = np.random.default_rng(seed)
        self.lr = lr
        self.W1 = rng.normal(0, 0.1, (n_input, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, 0.1, (n_hidden, 1))
        self.b2 = np.zeros(1)
        self.loss_hist = []

    @staticmethod
    def _relu(z):
        return np.maximum(z, 0)

    def _forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self._relu(self.z1)
        self.yhat = self.a1 @ self.W2 + self.b2
        return self.yhat

    def fit(self, X, y, epochs=200):
        X = np.asarray(X, float)
        y = np.asarray(y, float).reshape(-1, 1)
        n = len(X)
        for _ in range(epochs):
            yhat = self._forward(X)
            loss = np.mean((yhat - y) ** 2)
            self.loss_hist.append(loss)
            # 反向传播
            dy = 2 * (yhat - y) / n
            dW2 = self.a1.T @ dy
            db2 = dy.sum(axis=0)
            da1 = dy @ self.W2.T
            dz1 = da1 * (self.z1 > 0)
            dW1 = X.T @ dz1
            db1 = dz1.sum(axis=0)
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
        return self

    def predict(self, X):
        return self._forward(np.asarray(X, float)).ravel()
