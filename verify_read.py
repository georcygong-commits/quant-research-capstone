# -*- coding: utf-8 -*-
log = open(r"C:\Users\gucao\Desktop\Qlib\logs\verify_read_out.txt", "w", encoding="utf-8")
def out(*a):
    log.write(" ".join(str(x) for x in a) + "\n"); log.flush()

import qlib
qlib.init(provider_uri=r"C:\Users\gucao\Desktop\Qlib\data\qlib_data",
          region="cn", joblib_backend="threading")
from qlib.data import D

# 1) 分钟日历
cal = D.calendar(start_time="2026-09-01", end_time="2026-10-29", freq="1min")
out("calendar len:", len(cal), "first:", cal[0], "last:", cal[-1])

# 2) 股票池
inst = D.instruments(market="all")
codes = D.list_instruments(inst, as_list=True, freq="1min")
out("instruments:", codes)

# 3) 读取特征
df = D.features(["SZ000001"],
                ["$open", "$close", "$high", "$low", "$volume", "$factor"],
                start_time="2026-09-01", end_time="2026-10-29",
                freq="1min")
out("features shape:", df.shape)
out("head:\n", df.head(3).to_string())
out("tail:\n", df.tail(3).to_string())

# 4) 正确性核验：后复权价 / factor 应约等于不复权价（11.x 区间）
last = df.iloc[-1]
out("last bar:", dict(last))
out("close/factor =", last["$close"] / last["$factor"], "（应在真实股价 11 元附近）")

# 5) 缺失统计
out("NaN counts:\n", df.isna().sum().to_string())
# 6) 定位 NaN 行（判断是否为停牌）
nan_rows = df[df.isna().any(axis=1)]
out("NaN rows:\n", nan_rows.to_string())
log.close()
