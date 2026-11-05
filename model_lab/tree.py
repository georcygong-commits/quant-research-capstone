# -*- coding: utf-8 -*-
"""树模型：LightGBM（量化选股主力）。"""
import lightgbm as lgb
import pandas as pd


def fit_lgb(tr, va, feat, label="y_rank", num_leaves=31,
            learning_rate=0.05, n_estimators=500,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=200, seed=42):
    """训练 LightGBM，用验证集 early stopping。label 常用截面 rank 标签 y_rank。"""
    model = lgb.LGBMRegressor(
        num_leaves=num_leaves, learning_rate=learning_rate,
        n_estimators=n_estimators, subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_samples=min_child_samples,
        random_state=seed, verbosity=-1)
    model.fit(tr[feat], tr[label], eval_set=[(va[feat], va[label])],
              callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
    return model


def feature_importance(model, feat):
    return pd.Series(model.feature_importances_, index=feat).sort_values(
        ascending=False)
