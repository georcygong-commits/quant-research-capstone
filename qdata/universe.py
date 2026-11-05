# -*- coding: utf-8 -*-
"""全 A 股票池获取与代码格式转换。"""
import datetime
import pandas as pd
import loguru
import config
from qdata.wind import request

logger = loguru.logger

_EXCHANGE_MAP = {"SH": "sh", "SZ": "sz", "BJ": "bj"}


def wind_to_qlib(wind_code: str) -> str:
    """000001.SZ -> sz000001；600000.SH -> sh600000；430047.BJ -> bj430047。"""
    code, exch = wind_code.strip().split(".")
    return _EXCHANGE_MAP[exch.upper()] + code


def qlib_to_wind(qlib_code: str) -> str:
    """sz000001 -> 000001.SZ。"""
    q = qlib_code.strip().lower()
    exch, code = q[:2], q[2:]
    rev = {v: k for k, v in _EXCHANGE_MAP.items()}
    return f"{code}.{rev[exch]}"


def get_universe(date: str = None, force: bool = False) -> pd.DataFrame:
    """
    获取全 A 成分，返回 DataFrame[wind_code, qlib_code, name]，并缓存到 CSV。
    """
    if config.UNIVERSE_FILE.exists() and not force:
        df = pd.read_csv(config.UNIVERSE_FILE, dtype=str)
        if len(df) > 0:
            return df
    date = date or datetime.date.today().strftime("%Y-%m-%d")
    resp = request("wset", "sectorconstituent",
                   f"date={date};sectorid={config.ALL_A_SECTOR_ID}")
    # sectorconstituent 固定返回：date, wind_code, sec_name
    fields = [f.lower() for f in resp.Fields]
    code_idx = fields.index("wind_code")
    name_idx = fields.index("sec_name")
    codes = resp.Data[code_idx]
    names = resp.Data[name_idx]
    df = pd.DataFrame({
        "wind_code": codes,
        "qlib_code": [wind_to_qlib(c) for c in codes],
        "name": names,
    })
    df = df.drop_duplicates("wind_code").reset_index(drop=True)
    df.to_csv(config.UNIVERSE_FILE, index=False, encoding="utf-8-sig")
    logger.info(f"全 A 股票池 {len(df)} 只，已保存到 {config.UNIVERSE_FILE}")
    return df


def load_universe(limit: int = None) -> pd.DataFrame:
    """加载已缓存的股票池（必要时自动获取）。"""
    df = get_universe()
    if limit:
        df = df.iloc[:limit].reset_index(drop=True)
    return df


def get_index_constituent(index_code: str = None, date: str = None,
                          force: bool = False) -> pd.DataFrame:
    """
    获取指数成分（教学股票池），返回 DataFrame：
    [wind_code, qlib_code, name, industry, weight]，并缓存到 CSV。
    index_code 默认 config.TEACH_INDEX（沪深300）。
    """
    index_code = index_code or config.TEACH_INDEX
    safe = index_code.replace(".", "_").lower()
    out_file = config.INDEX_UNIVERSE_DIR / f"{safe}.csv"
    if out_file.exists() and not force:
        df = pd.read_csv(out_file, dtype=str)
        if len(df) > 0:
            return df
    date = date or datetime.date.today().strftime("%Y-%m-%d")
    resp = request("wset", "indexconstituent",
                   f"date={date};windcode={index_code}")
    fields = [f.lower() for f in resp.Fields]
    def col(name):
        return resp.Data[fields.index(name)] if name in fields else None
    codes = col("wind_code")
    names = col("sec_name")
    industry = col("industry")
    weight = col("i_weight")
    df = pd.DataFrame({
        "wind_code": codes,
        "qlib_code": [wind_to_qlib(c) for c in codes],
        "name": names,
        "industry": industry if industry is not None else "",
        "weight": weight if weight is not None else "",
    })
    df = df.drop_duplicates("wind_code").reset_index(drop=True)
    df.to_csv(out_file, index=False, encoding="utf-8-sig")
    logger.info(f"{index_code} 成分 {len(df)} 只，已保存到 {out_file}")
    return df


def load_teach_pool(limit: int = None) -> pd.DataFrame:
    """加载教学股票池（默认沪深300）。"""
    df = get_index_constituent()
    if limit:
        df = df.iloc[:limit].reset_index(drop=True)
    return df
