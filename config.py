# -*- coding: utf-8 -*-
"""
全局配置：路径、股票池、时间范围、频率、Wind 限频参数。
首次运行建议先用小样本（LIMIT_NSTOCKS）和短区间验证，再全量。
"""
from pathlib import Path

# ---------- 路径 ----------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UNIVERSE_DIR = DATA_DIR / "universe"
RAW_MINUTE_DIR = DATA_DIR / "raw_minute"      # 不复权原始分钟数据
ADJFACTOR_DIR = DATA_DIR / "adjfactor"        # 日频复权因子
QLIB_CSV_DIR = DATA_DIR / "qlib_csv"          # 组装后的后复权 CSV（喂给 dump_bin）
QLIB_DATA_DIR = DATA_DIR / "qlib_data"        # Qlib .bin 输出
STATE_DIR = BASE_DIR / "state"
LOG_DIR = BASE_DIR / "logs"
INDEX_UNIVERSE_DIR = DATA_DIR / "index_universe"   # 指数成分（沪深300等教学池）
RAW_DAILY_DIR = DATA_DIR / "raw_daily"             # 不复权原始日频
FUNDAMENTAL_DIR = DATA_DIR / "fundamental"         # 日频市值/估值（历史）
BENCHMARK_DIR = DATA_DIR / "benchmark"             # 基准指数日频
QLIB_DAILY_DIR = DATA_DIR / "qlib_daily"           # 日频 Qlib bin 输出
QLIB_DAILY_CSV_DIR = DATA_DIR / "qlib_daily_csv"   # 日频后复权 CSV（喂 dump）
for _d in (UNIVERSE_DIR, RAW_MINUTE_DIR, ADJFACTOR_DIR, QLIB_CSV_DIR,
           QLIB_DATA_DIR, STATE_DIR, LOG_DIR, INDEX_UNIVERSE_DIR,
           RAW_DAILY_DIR, FUNDAMENTAL_DIR, BENCHMARK_DIR, QLIB_DAILY_DIR,
           QLIB_DAILY_CSV_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------- Wind ----------
# 64 位 Wind 终端的 WindPy 所在目录（WindPy.pth 已写入 site-packages，这里仅作备用）
WIND_X64_DIR = r"D:\wind\WindNET\x64"
WIND_RETRY = 4          # 单次请求失败重试次数
WIND_SLEEP = 0.35       # 每次请求后的基础间隔（秒），用于规避频率限制
WIND_BATCH = 80         # wsd/wss 批量请求的股票只数

# ---------- 股票池 ----------
# 全部 A 股板块 ID（沪深京 A 股）
ALL_A_SECTOR_ID = "a001010100000000"
UNIVERSE_FILE = UNIVERSE_DIR / "all_a.csv"

# ---------- 频率与时间范围 ----------
FREQ = "1min"           # Qlib 频率标识
BAR_SIZE = "1"          # Wind wsi BarSize：1 分钟
# 分钟数据回溯起点（Wind 权限不同，回溯长度不同；2010 年后较完整）
START_DATE = "2023-01-01"
END_DATE = None         # None 表示到今天
CHUNK_MONTHS = 12       # wsi 单次分块的月数（年度分块，断点续传粒度）

# 交易时段（A 股），用于生成/校验时间
MORNING = ("09:30", "11:30")
AFTERNOON = ("13:00", "15:00")

# ---------- 字段 ----------
# 原始分钟字段（不复权）
MINUTE_FIELDS = "open,high,low,close,volume,amt"
# 组装后写入 Qlib 的字段（Qlib 至少需要 open/close/high/low/volume/factor）
QLIB_FIELDS = ["open", "close", "high", "low", "volume", "factor"]

# ---------- 调试开关 ----------
LIMIT_NSTOCKS = None    # 例如 5：只处理前 5 只股票；None = 全部

# ---------- 教学股票池：指数成分 ----------
HS300_INDEX = "000300.SH"   # 沪深300
ZZ500_INDEX = "000905.SH"   # 中证500
ZZ1000_INDEX = "000852.SH"  # 中证1000
TEACH_INDEX = HS300_INDEX   # 教学主线用沪深300

# ---------- 日频数据 ----------
DAILY_START_DATE = "2018-01-01"   # 日频因子研究起点
DAILY_FIELDS = "open,high,low,close,volume,amt"

# ---------- 基本面 / 风格（日频历史序列） ----------
FUNDAMENTAL_FIELDS = "mkt_cap_ard,mkt_cap_ashare,pe_ttm,pb_lf"

# ---------- 日频 Qlib 完整字段（基础 + 基本面） ----------
QLIB_DAILY_FIELDS = ["open", "close", "high", "low", "volume", "factor",
                     "mkt_cap_ard", "mkt_cap_ashare", "pe_ttm", "pb_lf"]
# 基本面字段名（与 fundamental 输出列一致，小写）
FUND_COLS = ["mkt_cap_ard", "mkt_cap_ashare", "pe_ttm", "pb_lf"]

# ---------- 基准指数 ----------
BENCHMARKS = ["000300.SH", "000905.SH", "000852.SH"]
