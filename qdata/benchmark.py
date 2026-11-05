# -*- coding: utf-8 -*-
"""
基准指数日频拉取：w.wsd（指数无需复权）。
存 data/benchmark/{index_safe}.csv。默认沪深300/中证500/中证1000。
"""
import datetime
import pandas as pd
import loguru
import config
from qdata.wind import request

logger = loguru.logger


def wsd_to_df(resp) -> pd.DataFrame:
    data = {f.lower(): resp.Data[i] for i, f in enumerate(resp.Fields)}
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(pd.Series(resp.Times))
    return df.reset_index(drop=True)


def _path(index_code: str):
    safe = index_code.replace(".", "_").lower()
    return config.BENCHMARK_DIR / f"{safe}.csv"


def fetch_one(index_code: str, start_date: str, end_date: str,
               force: bool = False) -> pd.DataFrame:
    """拉取单个指数日频，返回 DataFrame。"""
    path = _path(index_code)
    if path.exists() and not force:
        return pd.read_csv(path)

    existing = None
    if path.exists() and not force:
        existing = pd.read_csv(path)
        existing["date"] = pd.to_datetime(existing["date"])
    if force or existing is None or len(existing) == 0:
        start_d = start_date
    else:
        start_d = existing["date"].max().strftime("%Y-%m-%d")

    resp = request("wsd", index_code, config.DAILY_FIELDS,
                   start_d, end_date, "")
    new_df = wsd_to_df(resp)
    if existing is not None and len(existing):
        merged = (pd.concat([existing, new_df], ignore_index=True)
                  .drop_duplicates("date", keep="last")
                  .sort_values("date"))
    else:
        merged = new_df.drop_duplicates("date").sort_values("date")
    merged.to_csv(path, index=False, encoding="utf-8")
    return merged


def fetch_all(start_date: str = config.DAILY_START_DATE,
              end_date: str = None,
              force: bool = False):
    end_date = end_date or datetime.date.today().strftime("%Y-%m-%d")
    out = {}
    for idx in config.BENCHMARKS:
        out[idx] = fetch_one(idx, start_date, end_date, force=force)
        logger.info(f"基准 {idx}: {len(out[idx])} 行")
    logger.info("基准指数拉取结束。")
    return out
