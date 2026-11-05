# -*- coding: utf-8 -*-
"""步骤3：拉取日频复权因子（批量、增量）。"""
import sys
import pathlib
import argparse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from qdata.adjfactor import fetch_all

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    import config
    fetch_all(start_date=a.start or config.START_DATE, end_date=a.end,
              limit=a.limit, force=a.force)
