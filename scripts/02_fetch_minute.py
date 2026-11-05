# -*- coding: utf-8 -*-
"""步骤2：拉取全 A 1 分钟不复权数据（断点续传）。"""
import sys
import pathlib
import argparse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from qdata.minute import fetch_all

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--limit", type=int, default=None, help="只拉前 N 只（调试）")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    import config
    fetch_all(start_date=a.start or config.START_DATE, end_date=a.end,
              limit=a.limit, force=a.force)
