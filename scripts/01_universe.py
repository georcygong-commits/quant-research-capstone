# -*- coding: utf-8 -*-
"""步骤1：获取全 A 股票池。"""
import sys
import pathlib
import argparse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from qdata.universe import get_universe

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="强制刷新股票池")
    ap.add_argument("--date", default=None)
    a = ap.parse_args()
    df = get_universe(date=a.date, force=a.force)
    print(df.head(10).to_string())
    print("总数:", len(df))
