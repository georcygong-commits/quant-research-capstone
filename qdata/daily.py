# -*- coding: utf-8 -*-
"""
日频数据拉取：w.wsd（不复权 OHLCV），增量追加，去重写入。
原始数据存 data/raw_daily/{qlib_code}.csv。默认股票池为教学池（沪深300）。
"""
import datetime
import pandas as pd
from tqdm import tqdm
import loguru
import config
from qdata.wind import request
from qdata.universe import load_teach_pool, load_universe, qlib_to_wind

logger = loguru.logger


def wsd_to_df(resp) -> pd.DataFrame:
    """把 w.wsd 响应转成 DataFrame（字段小写，含 date 列）。"""
    data = {f.lower(): resp.Data[i] for i, f in enumerate(resp.Fields)}
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(pd.Series(resp.Times))
    # 剔除 OHLC 全空的行（无成交/占位）
    df = df.dropna(subset=["open", "high", "low", "close"], how="all")
    return df.reset_index(drop=True)


def _raw_path(qlib_code: str):
    return config.RAW_DAILY_DIR / f"{qlib_code}.csv"


def fetch_one(qlib_code: str, start_date: str, end_date: str,
              force: bool = False) -> int:
    """拉取单只股票日频数据（不复权），返回新增行数。"""
    wind_code = qlib_to_wind(qlib_code)
    path = _raw_path(qlib_code)
    end_d = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()

    existing = None
    if path.exists() and not force:
        existing = pd.read_csv(path)
        existing["date"] = pd.to_datetime(existing["date"])
    if force:
        start_d = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
    elif existing is not None and len(existing):
        # 从最后日期重拉（含最后一天，便于修正当日数据）
        start_d = existing["date"].max().date()
    else:
        start_d = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()

    b = start_d.strftime("%Y-%m-%d")
    e = end_d.strftime("%Y-%m-%d")
    resp = request("wsd", wind_code, config.DAILY_FIELDS, b, e, "")
    new_df = wsd_to_df(resp)
    if new_df.empty:
        return 0
    if existing is not None and len(existing):
        merged = (pd.concat([existing, new_df], ignore_index=True)
                  .drop_duplicates("date", keep="last")
                  .sort_values("date"))
    else:
        merged = new_df.drop_duplicates("date").sort_values("date")
    merged.to_csv(path, index=False, encoding="utf-8")
    return len(merged) - (len(existing) if existing is not None else 0)


def fetch_all(start_date: str = config.DAILY_START_DATE,
              end_date: str = None,
              limit: int = None,
              pool: str = "teach",
              force: bool = False):
    """
    拉取日频数据。
    pool: 'teach' 沪深300教学池；'all' 全 A。
    """
    end_date = end_date or datetime.date.today().strftime("%Y-%m-%d")
    if pool == "teach":
        uni = load_teach_pool(limit)
    else:
        uni = load_universe(limit or config.LIMIT_NSTOCKS)
    logger.info(f"开始拉取日频数据（{pool}）：{len(uni)} 只，{start_date} ~ {end_date}")
    failed = {}
    for _, row in tqdm(list(uni.iterrows()), desc="日频数据"):
        qc = row["qlib_code"]
        try:
            fetch_one(qc, start_date, end_date, force=force)
        except Exception as e:
            failed[qc] = repr(e)
            logger.warning(f"{qc} 拉取失败: {e}")
    if failed:
        logger.warning(f"共 {len(failed)} 只失败，重跑即可: {list(failed)[:10]}")
    logger.info("日频数据拉取结束。")
    return failed
