# -*- coding: utf-8 -*-
"""
组装后复权 CSV（喂给 dump_bin）：
  不复权分钟价 × 当日复权因子 = 后复权分钟价；volume 保持原始。
输出 data/qlib_csv/{qlib_code}.csv，列 date,symbol,open,close,high,low,volume,factor。
"""
import subprocess
import sys
import pandas as pd
from tqdm import tqdm
import loguru
import config
from qdata.universe import load_universe

logger = loguru.logger


def build_one(qlib_code: str):
    m_path = config.RAW_MINUTE_DIR / f"{qlib_code}.csv"
    a_path = config.ADJFACTOR_DIR / f"{qlib_code}.csv"
    if not m_path.exists() or not a_path.exists():
        return False
    m = pd.read_csv(m_path)
    a = pd.read_csv(a_path)
    m["date"] = pd.to_datetime(m["date"])
    a["date"] = pd.to_datetime(a["date"])
    m["day"] = m["date"].dt.strftime("%Y-%m-%d")
    a["day"] = a["date"].dt.strftime("%Y-%m-%d")
    df = m.merge(a[["day", "factor"]], on="day", how="left")
    # 因子缺失（新股/未覆盖）向前填充，仍缺则因子=1（等价不复权）
    df["factor"] = df["factor"].ffill().bfill().fillna(1.0)
    # 剔除收盘集合竞价占位行（OHLC 原始全空，如 14:58/14:59）
    df = df.dropna(subset=["open", "high", "low", "close"], how="all")

    for f in ("open", "high", "low", "close"):
        df[f] = df[f] * df["factor"]

    out = pd.DataFrame({
        "date": df["date"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": qlib_code,
        "open": df["open"], "close": df["close"],
        "high": df["high"], "low": df["low"],
        "volume": df["volume"], "factor": df["factor"],
    })
    out = out.drop_duplicates("date").sort_values("date")
    out.to_csv(config.QLIB_CSV_DIR / f"{qlib_code}.csv", index=False,
               encoding="utf-8")
    return True


def build_all(limit: int = None):
    uni = load_universe(limit or config.LIMIT_NSTOCKS)
    ok = 0
    for qc in tqdm(uni["qlib_code"], desc="组装后复权"):
        if build_one(qc):
            ok += 1
    logger.info(f"组装完成：{ok}/{len(uni)} 只，输出到 {config.QLIB_CSV_DIR}")
    return ok


# ===================== 日频组装 =====================

def build_daily_one(qlib_code: str):
    """日频后复权组装：不复权日频 × adjfactor，合并基本面字段。"""
    d_path = config.RAW_DAILY_DIR / f"{qlib_code}.csv"
    a_path = config.ADJFACTOR_DIR / f"{qlib_code}.csv"
    f_path = config.FUNDAMENTAL_DIR / f"{qlib_code}.csv"
    if not d_path.exists() or not a_path.exists():
        return False
    d = pd.read_csv(d_path)
    a = pd.read_csv(a_path)
    d["date"] = pd.to_datetime(d["date"])
    a["date"] = pd.to_datetime(a["date"])
    d["day"] = d["date"].dt.strftime("%Y-%m-%d")
    a["day"] = a["date"].dt.strftime("%Y-%m-%d")
    df = d.merge(a[["day", "factor"]], on="day", how="left")
    df["factor"] = df["factor"].ffill().bfill().fillna(1.0)

    # 基本面（若存在）
    if f_path.exists():
        f = pd.read_csv(f_path)
        f["date"] = pd.to_datetime(f["date"])
        f["day"] = f["date"].dt.strftime("%Y-%m-%d")
        have = [c for c in config.FUND_COLS if c in f.columns]
        df = df.merge(f[["day"] + have], on="day", how="left")
        for c in have:
            df[c] = df[c].ffill()

    # OHLC 后复权
    for c in ("open", "high", "low", "close"):
        df[c] = df[c] * df["factor"]
    df = df.dropna(subset=["open", "high", "low", "close"], how="all")

    out = pd.DataFrame({"date": df["day"], "symbol": qlib_code})
    for c in config.QLIB_DAILY_FIELDS:
        if c in df.columns:
            out[c] = df[c]
    out = out.drop_duplicates("date").sort_values("date")
    out.to_csv(config.QLIB_DAILY_CSV_DIR / f"{qlib_code}.csv", index=False,
               encoding="utf-8")
    return True


def build_daily_all(limit: int = None, pool: str = "teach"):
    from qdata.universe import load_teach_pool
    if pool == "teach":
        uni = load_teach_pool(limit)
    else:
        uni = load_universe(limit or config.LIMIT_NSTOCKS)
    ok = 0
    for qc in tqdm(uni["qlib_code"], desc="组装日频后复权"):
        if build_daily_one(qc):
            ok += 1
    logger.info(f"日频组装完成：{ok}/{len(uni)} 只，输出到 {config.QLIB_DAILY_CSV_DIR}")
    return ok


def dump_daily_bin(mode: str = "all"):
    """调用 dump_bin.py 生成日频 Qlib .bin。"""
    script = config.BASE_DIR / "qdata" / "dump_bin.py"
    cmd = [
        sys.executable, str(script), f"dump_{mode}",
        "--data_path", str(config.QLIB_DAILY_CSV_DIR),
        "--qlib_dir", str(config.QLIB_DAILY_DIR),
        "--freq", "day",
        "--date_field_name", "date",
        "--symbol_field_name", "symbol",
        "--include_fields", ",".join(config.QLIB_DAILY_FIELDS),
        "--file_suffix", ".csv",
    ]
    logger.info(" ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        logger.error(r.stdout); logger.error(r.stderr)
        raise RuntimeError("日频 dump_bin 失败")
    logger.info("日频 dump_bin 完成：" + str(config.QLIB_DAILY_DIR))


def dump_bin(mode: str = "all"):
    """调用项目内 dump_bin.py 生成 Qlib .bin。mode: all / update。"""
    script = config.BASE_DIR / "qdata" / "dump_bin.py"
    cmd = [
        sys.executable, str(script),
        f"dump_{mode}",
        "--data_path", str(config.QLIB_CSV_DIR),
        "--qlib_dir", str(config.QLIB_DATA_DIR),
        "--freq", config.FREQ,
        "--date_field_name", "date",
        "--symbol_field_name", "symbol",
        "--include_fields", ",".join(config.QLIB_FIELDS),
        "--file_suffix", ".csv",
    ]
    logger.info(" ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        logger.error(r.stdout)
        logger.error(r.stderr)
        raise RuntimeError("dump_bin 失败")
    logger.info("dump_bin 完成：" + str(config.QLIB_DATA_DIR))
