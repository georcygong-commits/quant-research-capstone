# -*- coding: utf-8 -*-
"""集成学习与校准：投票/Stacking、加权融合、概率校准。"""
import numpy as np
import pandas as pd


def make_base_models(seed=42):
    """一组异构基模型。"""
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    import lightgbm as lgb
    return {
        "lgb": lgb.LGBMRegressor(n_estimators=150, num_leaves=31,
                                  learning_rate=0.05, random_state=seed,
                                  verbose=-1),
        "ridge": Ridge(alpha=10.0),
        "mlp": MLPRegressor(hidden_layer_sizes=(32,), max_iter=200,
                            random_state=seed, early_stopping=True),
    }


def build_voting(models):
    """简单平均投票集成。"""
    from sklearn.ensemble import VotingRegressor
    return VotingRegressor([(n, m) for n, m in models.items()])


def build_stacking(models, meta=None):
    """Stacking：基模型 out-of-fold 预测喂给元模型。"""
    from sklearn.ensemble import StackingRegressor
    from sklearn.linear_model import Ridge
    if meta is None:
        meta = Ridge(alpha=1.0)
    return StackingRegressor(estimators=[(n, m) for n, m in models.items()],
                             final_estimator=meta,
                             passthrough=False)


def weighted_blend(pred_df, weights):
    """自实现加权融合：pred_df 列为各模型预测，weights 为权重字典。"""
    w = pd.Series(weights)
    w = w / w.sum()
    return pred_df[w.index].mul(w, axis=1).sum(axis=1)


def calibration_curve(y_true, y_pred, n_bins=10):
    """校准曲线：按预测分箱，比较箱内预测均值与真实均值。"""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    edges = np.quantile(y_pred, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    rows = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            msk = (y_pred >= lo) & (y_pred <= hi)
        else:
            msk = (y_pred >= lo) & (y_pred < hi)
        if msk.sum() > 0:
            rows.append({"bin": i, "n": int(msk.sum()),
                         "pred_mean": y_pred[msk].mean(),
                         "true_mean": y_true[msk].mean()})
    return pd.DataFrame(rows)


def fit_isotonic(y_true, y_pred):
    """保序回归校准器。"""
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(np.asarray(y_pred, float), np.asarray(y_true, float))
    return ir
