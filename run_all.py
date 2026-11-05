# -*- coding: utf-8 -*-
"""
一键全流程：
  1 获取全A股票池
  2 拉取分钟数据（不复权，断点续传）
  3 拉取复权因子
  4 组装后复权 CSV
  5 dump 成 Qlib .bin

首次建议：python run_all.py --limit 3 --start 2026-09-01   （小样本验证）
全量：    python run_all.py
"""
import argparse
import loguru
import config
from qdata.universe import get_universe
from qdata.minute import fetch_all as fetch_minute
from qdata.adjfactor import fetch_all as fetch_adj
from qdata.build import build_all, dump_bin

logger = loguru.logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=config.START_DATE)
    ap.add_argument("--end", default=None)
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 只（调试）")
    ap.add_argument("--steps", default="1,2,3,4,5", help="要执行的步骤，如 1,2,3")
    ap.add_argument("--dump-mode", default="all", choices=["all", "update"])
    a = ap.parse_args()
    steps = {int(s) for s in a.steps.split(",")}

    if 1 in steps:
        get_universe(force=False)
    if 2 in steps:
        fetch_minute(start_date=a.start, end_date=a.end, limit=a.limit)
    if 3 in steps:
        fetch_adj(start_date=a.start, end_date=a.end, limit=a.limit)
    if 4 in steps:
        build_all(limit=a.limit)
    if 5 in steps:
        dump_bin(mode=a.dump_mode)
    logger.info("全部完成。Qlib 数据目录：" + str(config.QLIB_DATA_DIR))


if __name__ == "__main__":
    main()
