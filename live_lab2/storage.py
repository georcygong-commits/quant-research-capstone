# -*- coding: utf-8 -*-
"""
时序/关系数据库与数据模型（M6.5）：
用 SQLite 模拟时序库，实现 Point-in-Time（PIT）存储——
查询时只能看到当时已经入库的数据，从存储层杜绝未来函数。
"""
import sqlite3
import pandas as pd


class PITStore:
    """
    PIT 存储：每条观测带业务时间 ts 与入库时间 ingest_ts。
    写入幂等（同主键重复写覆盖），asof 查询只返回当时已知数据。
    """
    def __init__(self, path=":memory:"):
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS observations(
                   code TEXT, ts TEXT, field TEXT, value REAL,
                   ingest_ts TEXT, PRIMARY KEY(code, ts, field))""")

    def upsert(self, code, ts, field, value, ingest_ts):
        """幂等写入：相同 (code,ts,field) 重复写则覆盖。"""
        self.conn.execute(
            "INSERT OR REPLACE INTO observations VALUES(?,?,?,?,?)",
            (code, ts, field, float(value), ingest_ts))
        self.conn.commit()

    def asof_query(self, code, field, asof):
        """PIT 查询：业务时间与入库时间都不晚于 asof 的最新值。"""
        cur = self.conn.execute(
            """SELECT ts, value FROM observations
               WHERE code=? AND field=? AND ts<=? AND ingest_ts<=?
               ORDER BY ts DESC LIMIT 1""", (code, field, asof, asof))
        row = cur.fetchone()
        return None if row is None else {"ts": row[0], "value": row[1]}

    def history(self, code, field, asof):
        """返回 asof 时点可见的完整历史（DataFrame）。"""
        return pd.read_sql(
            """SELECT ts, value FROM observations
               WHERE code=? AND field=? AND ts<=? AND ingest_ts<=?
               ORDER BY ts""", self.conn, params=(code, field, asof, asof))

    def close(self):
        self.conn.close()
