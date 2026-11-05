# -*- coding: utf-8 -*-
"""
日频教学数据一键流程（默认沪深300）：
  1 获取沪深300成分（含 Wind 行业）
  2 拉取复权因子（教学池）
  3 拉取日频 OHLCV（不复权）
  4 拉取基本面（市值/估值）
  5 拉取基准指数（沪深300/中证500/中证1000）
  6 组装日频后复权 CSV 并 dump 成 Qlib 日频 bin

小样本：python run_daily.py --limit 20
全量  ：python run_daily.py
"""
import argparse
import loguru
import config
from qdata.universe import get_index_constituent
from qdata.adjfactor import fetch_all as fetch_adj
from qdata.daily import fetch_all as fetch_daily
from qdata.fundamental import fetch_all as fetch_fund
from qdata.benchmark import fetch_all as fetch_bench
from qdata.build import build_daily_all, dump_daily_bin

logger = loguru.logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=config.DAILY_START_DATE)
    ap.add_argument("--end", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--index", default=config.TEACH_INDEX)
    ap.add_argument("--steps", default="1,2,3,4,5,6")
    ap.add_argument("--dump-mode", default="all", choices=["all", "update"])
    a = ap.parse_args()
    steps = {int(s) for s in a.steps.split(",")}

    if 1 in steps:
        get_index_constituent(a.index, force=False)
    if 2 in steps:
        fetch_adj(start_date=a.start, end_date=a.end, limit=a.limit, pool="teach")
    if 3 in steps:
        fetch_daily(start_date=a.start, end_date=a.end, limit=a.limit, pool="teach")
    if 4 in steps:
        fetch_fund(start_date=a.start, end_date=a.end, limit=a.limit, pool="teach")
    if 5 in steps:
        fetch_bench(start_date=a.start, end_date=a.end)
    if 6 in steps:
        build_daily_all(limit=a.limit, pool="teach")
        dump_daily_bin(mode=a.dump_mode)
    logger.info("全部完成。日频 Qlib 数据目录：" + str(config.QLIB_DAILY_DIR))


if __name__ == "__main__":
    main()
