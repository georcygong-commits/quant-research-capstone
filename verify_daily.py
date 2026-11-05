# -*- coding: utf-8 -*-
"""日频数据健康检查：回读 qlib_daily，验证后复权、factor、基本面。"""
import sys
sys.path.insert(0, r"C:\Users\gucao\Desktop\Qlib")
import qlib
from qlib.data import D
import config

qlib.init(provider_uri=str(config.QLIB_DAILY_DIR), joblib_backend="threading")
cal = D.calendar(freq="day")
inst = D.instruments(market="all")
codes = D.list_instruments(instruments=inst, as_list=True, freq="day")
print("calendar:", len(cal), cal[0], "~", cal[-1])
print("instruments:", len(codes), codes[:5])

fields = ["$open", "$close", "$high", "$low", "$volume", "$factor",
          "$mkt_cap_ard", "$mkt_cap_ashare", "$pe_ttm", "$pb_lf"]
df = D.features(codes, fields, freq="day")
print("features shape:", df.shape)

# 验证一只：后复权/factor = 真实股价
for code in codes[:2]:
    s = df.xs(code, level="instrument").dropna(subset=["$close"])
    if len(s) == 0:
        continue
    last = s.tail(1)
    c, f = float(last["$close"].iloc[0]), float(last["$factor"].iloc[0])
    print(f"\n{code}: 后复权close={c:.2f}, factor={f:.3f}, close/factor={c/f:.2f}")
    print(f"  总市值={float(last['$mkt_cap_ard'].iloc[0])/1e8:.0f}亿, "
          f"PE={float(last['$pe_ttm'].iloc[0]):.2f}, PB={float(last['$pb_lf'].iloc[0]):.2f}")

print("\nNaN counts:")
print(df.isna().sum().to_string())
