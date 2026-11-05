# -*- coding: utf-8 -*-
"""
分钟数据拉取：w.wsi（不复权），按年度分块，断点续传，去重写入。
原始数据存 data/raw_minute/{qlib_code}.csv。
"""
import json
import datetime
import pandas as pd
from tqdm import tqdm
import loguru
import config
from qdata.wind import request
from qdata.universe import load_universe, qlib_to_wind

logger = loguru.logger
STATE_FILE = config.STATE_DIR / "fetch_minute_state.json"


def _load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {}


def _save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2),
                          encoding="utf-8")


def wsi_to_df(resp) -> pd.DataFrame:
    """把 w.wsi 响应转成 DataFrame（字段小写，含 date 列）。"""
    data = {f.lower(): resp.Data[i] for i, f in enumerate(resp.Fields)}
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(pd.Series(resp.Times))
    # 剔除收盘集合竞价占位行（14:58/14:59 等 OHLC 全空、volume=0 的无信息行）
    df = df.dropna(subset=["open", "high", "low", "close"], how="all")
    return df.reset_index(drop=True)


def month_chunks(start: datetime.date, end: datetime.date, months: int):
    """按自然月数生成不重叠的 (beg_date, end_date) 日历日区间。"""
    cur = start
    while cur <= end:
        nxt = (pd.Timestamp(cur) + pd.DateOffset(months=months)).date()
        chunk_end = min(nxt - datetime.timedelta(days=1), end)
        yield cur, chunk_end
        cur = nxt


def _raw_path(qlib_code: str) -> "pathlib.Path":
    return config.RAW_MINUTE_DIR / f"{qlib_code}.csv"


def fetch_one(qlib_code: str, start_date: str, end_date: str,
              state: dict, force: bool = False) -> int:
    """拉取单只股票分钟数据，返回新增行数。"""
    wind_code = qlib_to_wind(qlib_code)
    path = _raw_path(qlib_code)
    end_d = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()

    # 已有数据：确定续传起点
    existing = None
    if path.exists() and not force:
        existing = pd.read_csv(path)
        existing["date"] = pd.to_datetime(existing["date"])
    if force:
        start_d = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
    elif existing is not None and len(existing):
        # 从最后一根 bar 的次日开始；若最后 bar 在非交易时段，wsi 会自动处理
        last_ts = existing["date"].max()
        start_d = last_ts.date()
        # 若最后日期就是今天（盘中），重拉今天以补全
        if start_d == end_d:
            existing = existing[existing["date"].dt.date < end_d]
    else:
        start_d = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()

    total_new = 0
    for cbeg, cend in month_chunks(start_d, end_d, config.CHUNK_MONTHS):
        b = cbeg.strftime("%Y-%m-%d") + " 09:30:00"
        e = cend.strftime("%Y-%m-%d") + " 15:00:00"
        resp = request("wsi", wind_code, config.MINUTE_FIELDS, b, e,
                       f"BarSize={config.BAR_SIZE}")
        new_df = wsi_to_df(resp)
        if not new_df.empty:
            if existing is not None and len(existing):
                merged = (pd.concat([existing, new_df], ignore_index=True)
                          .drop_duplicates("date", keep="last")
                          .sort_values("date"))
            else:
                merged = new_df.drop_duplicates("date").sort_values("date")
            merged.to_csv(path, index=False, encoding="utf-8")
            existing = merged
            total_new += len(new_df)
            state[qlib_code] = str(merged["date"].max())
            _save_state(state)
    return total_new


def fetch_all(start_date: str = config.START_DATE,
              end_date: str = None,
              limit: int = None,
              force: bool = False):
    end_date = end_date or datetime.date.today().strftime("%Y-%m-%d")
    uni = load_universe(limit or config.LIMIT_NSTOCKS)
    state = _load_state()
    logger.info(f"开始拉取分钟数据：{len(uni)} 只，{start_date} ~ {end_date}")
    failed = {}
    for _, row in tqdm(list(uni.iterrows()), desc="分钟数据"):
        qc = row["qlib_code"]
        try:
            fetch_one(qc, start_date, end_date, state, force=force)
        except Exception as e:
            failed[qc] = repr(e)
            logger.warning(f"{qc} 拉取失败: {e}")
    _save_state(state)
    if failed:
        logger.warning(f"共 {len(failed)} 只失败，重跑即可续传: {list(failed)[:10]}")
    logger.info("分钟数据拉取结束。")
    return failed
