# -*- coding: utf-8 -*-
"""
复权因子拉取：w.wsd adjfactor（日频，定点累计、历史不变），批量 + 增量。
存 data/adjfactor/{qlib_code}.csv，列 date, factor。
"""
import datetime
import pandas as pd
from tqdm import tqdm
import loguru
import config
from qdata.wind import request
from qdata.universe import load_universe, load_teach_pool, wind_to_qlib

logger = loguru.logger


def _path(qlib_code: str):
    return config.ADJFACTOR_DIR / f"{qlib_code}.csv"


def fetch_all(start_date: str = config.START_DATE,
              end_date: str = None,
              limit: int = None,
              pool: str = "all",
              force: bool = False):
    end_date = end_date or datetime.date.today().strftime("%Y-%m-%d")
    if pool == "teach":
        uni = load_teach_pool(limit)
    else:
        uni = load_universe(limit or config.LIMIT_NSTOCKS)
    codes = uni["wind_code"].tolist()
    end_d = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()

    # 按批量分组
    groups = [codes[i:i + config.WIND_BATCH]
              for i in range(0, len(codes), config.WIND_BATCH)]
    logger.info(f"开始拉取复权因子：{len(codes)} 只，{start_date} ~ {end_date}")
    failed = {}

    for grp in tqdm(groups, desc="复权因子"):
        # 组内取最早的续传起点（简单起见：组内按各自文件判断，缺者用 start_date）
        begins = {}
        for wc in grp:
            qc = wind_to_qlib(wc)
            p = _path(qc)
            if force or not p.exists():
                begins[wc] = start_date
            else:
                old = pd.read_csv(p)
                if len(old):
                    begins[wc] = (pd.to_datetime(old["date"]).max().date()
                                  + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
                else:
                    begins[wc] = start_date
        g_start = min(begins.values())
        if g_start > end_date:
            continue
        try:
            resp = request("wsd", ",".join(grp), "adjfactor", g_start, end_date, "")
        except Exception as e:
            for wc in grp:
                failed[wind_to_qlib(wc)] = repr(e)
            continue

        times = pd.to_datetime(pd.Series(resp.Times)).dt.strftime("%Y-%m-%d").tolist()
        for i, wc in enumerate(resp.Codes):
            qc = wind_to_qlib(wc)
            p = _path(qc)
            new_df = pd.DataFrame({"date": times, "factor": resp.Data[i]})
            # 只保留该股票实际起点之后
            new_df = new_df[new_df["date"] >= begins[wc]]
            new_df = new_df.dropna(subset=["factor"])
            if p.exists() and not force:
                old = pd.read_csv(p)
                out = (pd.concat([old, new_df], ignore_index=True)
                       .drop_duplicates("date", keep="last")
                       .sort_values("date"))
            else:
                out = new_df.drop_duplicates("date").sort_values("date")
            out.to_csv(p, index=False, encoding="utf-8")

    if failed:
        logger.warning(f"{len(failed)} 只复权因子失败，重跑即可: {list(failed)[:10]}")
    logger.info("复权因子拉取结束。")
    return failed
