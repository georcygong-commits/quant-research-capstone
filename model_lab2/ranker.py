# -*- coding: utf-8 -*-
"""排序学习：LightGBM LambdaRank 与 NDCG 评估。"""
import numpy as np
import pandas as pd


def dcg(rels, k):
    rels = np.asarray(rels, float)[:k]
    if len(rels) == 0:
        return 0.0
    gain = 2 ** rels - 1
    discount = np.log2(np.arange(2, len(rels)+2))
    return np.sum(gain / discount)


def ndcg_at_k(true_rel, pred_score, k=10):
    """单组 NDCG@k：按预测排序，与理想排序对比。"""
    order = np.argsort(-np.asarray(pred_score, float))
    ret = np.asarray(true_rel, float)
    dcg_p = dcg(ret[order], k)
    dcg_i = dcg(np.sort(ret)[::-1], k)
    return dcg_p / dcg_i if dcg_i > 0 else np.nan


def make_relevance(label, n_level=5):
    """把连续标签按截面分桶成 0~n_level-1 的相关性等级。"""
    return label.apply(lambda s: pd.qcut(s, n_level, labels=False,
                                         duplicates="drop"), axis=1)


def train_lgbm_ranker(train_df, feature_names, rel_col="rel"):
    """训练 LGBMRanker，按日期分组。"""
    import lightgbm as lgb
    model = lgb.LGBMRanker(objective="lambdarank", n_estimators=200,
                           learning_rate=0.05, num_leaves=31,
                           random_state=42, verbose=-1)
    groups = train_df.groupby(level="datetime").size().values
    model.fit(train_df[feature_names], train_df[rel_col], group=groups)
    return model


def evaluate_ndcg(model, test_df, feature_names, rel_col="rel", k=10):
    """逐日期计算平均 NDCG@k。"""
    scores = []
    for _, sub in test_df.groupby(level="datetime"):
        if len(sub) > k:
            pred = model.predict(sub[feature_names])
            scores.append(ndcg_at_k(sub[rel_col].values, pred, k))
    return pd.Series(scores)
