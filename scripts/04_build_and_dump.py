# -*- coding: utf-8 -*-
"""步骤4：组装后复权 CSV 并 dump 成 Qlib .bin。"""
import sys
import pathlib
import argparse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from qdata.build import build_all, dump_bin

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--mode", default="all", choices=["all", "update"])
    ap.add_argument("--no-dump", action="store_true", help="只组装，不 dump")
    a = ap.parse_args()
    build_all(limit=a.limit)
    if not a.no_dump:
        dump_bin(mode=a.mode)
