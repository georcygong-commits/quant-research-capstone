# -*- coding: utf-8 -*-
"""
L2 公共配置：在 L1 config 基础上，提供参数化股票池与随机种子。
所有 L2 研究代码通过 load_universe(name) 切换 hs300 / zz500 / allA，
保证同一套逻辑、三套配置与产出。
"""
import os, random
import numpy as np
import pandas as pd
import config as l1

# 固定随机种子，保证全流程可复现
SEED = 42

# 股票池注册表：名称 -> 成分文件（相对 data）与对应指数
UNIVERSES = {
    "hs300": {"index": "000300.SH",
              "file": l1.INDEX_UNIVERSE_DIR / "000300_sh.csv",
              "data_ready": True},
    "zz500": {"index": "000905.SH",
              "file": l1.INDEX_UNIVERSE_DIR / "000905_sh.csv",
              "data_ready": False},
    "allA": {"index": None,
             "file": l1.UNIVERSE_DIR / "all_a.csv",
             "data_ready": False},
}


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
    return seed


def load_universe(name="hs300", upper=True):
    """
    返回指定股票池的 qlib_code 列表（默认大写以对齐 Qlib 数据）。
    name: hs300 / zz500 / allA。
    """
    if name not in UNIVERSES:
        raise KeyError("未知股票池 %s，可选 %s" % (name, list(UNIVERSES)))
    meta = UNIVERSES[name]
    f = meta["file"]
    if not os.path.exists(f):
        raise FileNotFoundError(
            "股票池 %s 的成分文件尚未就绪：%s；需先在数据工程中下载。" % (name, f))
    df = pd.read_csv(f)
    codes = df["qlib_code"].astype(str)
    if upper:
        codes = codes.str.upper()
    return codes.tolist()


def universe_status():
    """返回各股票池成分文件与数据就绪状态。"""
    out = {}
    for name, meta in UNIVERSES.items():
        out[name] = {
            "file_exists": os.path.exists(meta["file"]),
            "qlib_data_ready": meta["data_ready"],
            "index": meta["index"],
        }
    return out
