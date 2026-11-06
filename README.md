# A 股量化研究 · 端到端作品集

以 **Microsoft Qlib** 为底座，系统覆盖量化研究全链路：数据工程 → 因子研究 → 机器学习建模 →
组合与风险管理 → 回测归因 → 实盘工程 → 横向策略，并沉淀为一个自包含、可复现的端到端多因子选股项目。

## 核心成果（Capstone）

端到端多因子选股策略（沪深 300，300 只，含交易成本）：

| 指标 | 策略 | 基准 |
|---|---|---|
| 年化收益 | 16.00% | 14.85% |
| 夏普比率 | 1.01 | 0.80 |
| 最大回撤 | -24.01% | -26.12% |
| 样本外 RankIC | 0.0143（胜率 53.6%） | — |

详见 [`capstone/`](capstone/README.md)（含 notebook、复现说明、面试题库、简历陈述与模拟面试记录）。

## 能力地图

| 模块 | 内容 | 代码包 |
|---|---|---|
| 微观结构 | 订单簿、价差、Kyle/Amihud、价格冲击 | `micro_lab` |
| 因子研究 | 量价/基本面因子、Barra、Fama-MacBeth、因子挖掘 | `factor_lab2` |
| 模型预测 | LightGBM、Ranker、深度/强化学习、滚动验证、集成 | `model_lab2` |
| 组合风险 | 协方差收缩、GARCH/DCC、风险平价、BL、极值对冲 | `port_lab2` |
| 回测归因 | 事件驱动、TCA、Brinson、过拟合检验、容量 | `bt_lab2` |
| 实盘工程 | 实时行情风控、自适应执行、存储部署、监控运维 | `live_lab2` |
| 横向策略 | CTA、期权、波动率曲面、配对套利、做市 | `deriv_lab` |

## 模拟盘与一致性跟踪

回测与模拟共用同一信号引擎，从 2024-01-31 起逐日跟踪，月末调仓、日频记录。
663 个交易日的对账显示：调仓目标权重完全一致（3.47e-17），稳态日收益/净值差异达机器精度
（4.5e-14），总体结论 **PASS**；跟踪区间模拟盘年化 24.79%、夏普 1.50、最大回撤 -11.78%。
详见 [`paper_trading/`](paper_trading/README.md)。

## 仓库导航

- `capstone/`：端到端策略作品集（主交付）
- `paper_trading/`：模拟盘与回测/实盘一致性跟踪
- `notebooks/`：分模块教学 notebook（含 HTML 渲染）
- `l2/`：公共配置与参数化股票池
- 各 `*_lab*` 目录：可复用的研究模块
- `config.py` / `run_all.py` / `run_daily.py`：全局配置与数据工程入口

## 快速开始

```powershell
conda activate qlib
jupyter nbconvert --to notebook --execute --inplace `
  capstone\capstone_strategy.ipynb --ExecutePreprocessor.kernel_name=qlib
```

---

# 附：Wind → Qlib 全 A 分钟数据全流程

基于 **WindPy** 拉取全 A 股 1 分钟数据，按复权因子计算后复权价，转换为 **Microsoft Qlib** 的 `.bin` 格式，供因子挖掘、模型训练与回测使用。全流程支持断点续传与增量更新。

---

## 一、环境准备

### 1. 创建 conda 环境（已完成可跳过）

```powershell
conda create -n qlib python=3.10 -y
conda activate qlib
pip install pyqlib pandas numpy tqdm loguru fire
```

当前环境已安装：`qlib 0.9.7 / pandas 2.3.3 / numpy 2.2.6`。

### 2. 配置 WindPy

- Wind 终端安装目录：`D:\wind\WindNET`（64 位接口在 `D:\wind\WindNET\x64`）。
- 已在环境的 `site-packages` 下写入 `WindPy.pth`，内容为 `D:\wind\WindNET\x64`，用于加载 `WindPy.dll`。
- 使用前需 **启动并登录 Wind 终端**。

验证：

```powershell
python -c "from WindPy import w; w.start(); print(w.isconnected())"
```

---

## 二、目录结构

```
Qlib/
├── config.py               # 全局配置（路径、股票池、时间范围、限频、字段）
├── run_all.py              # 一键全流程（支持 --limit/--start/--steps）
├── verify_read.py          # 数据健康检查（Qlib 回读验证）
├── qdata/
│   ├── wind.py             # Wind 连接、重连、限频、重试封装
│   ├── universe.py         # 全 A 股票池获取与代码转换
│   ├── minute.py           # w.wsi 分钟数据拉取（年度分块、断点续传）
│   ├── adjfactor.py        # w.wsd 复权因子拉取（批量、增量）
│   ├── build.py            # 组装后复权 CSV + 调用 dump_bin
│   └── dump_bin.py         # 官方 CSV→bin 转换脚本
├── scripts/                # 分步入口 01~04
├── data/
│   ├── universe/all_a.csv  # 全 A 股票池（当前 5577 只）
│   ├── raw_minute/         # 不复权原始分钟数据（每只一个 CSV）
│   ├── adjfactor/          # 日频复权因子（每只一个 CSV）
│   ├── qlib_csv/           # 组装后的后复权 CSV
│   └── qlib_data/          # ★ Qlib .bin 输出（calendars/instruments/features）
├── state/                  # 断点续传状态
└── logs/
```

---

## 三、全流程说明

| 步骤 | 内容 | 接口 |
| --- | --- | --- |
| 1 | 获取全 A 股票池（板块 `a001010100000000`） | `w.wset` |
| 2 | 拉取不复权 1 分钟 OHLCV、成交额，按年度分块 | `w.wsi` |
| 3 | 拉取日频复权因子（批量 80 只/次） | `w.wsd` |
| 4 | 组装：后复权价 = 不复权价 × 当日复权因子 | 本地计算 |
| 5 | dump 成 Qlib `.bin`（`calendars/1min.txt`、`features/{code}/*.1min.bin`） | `dump_bin.py` |

### 代码格式

- Wind：`000001.SZ` / `600000.SH` / `430047.BJ`
- Qlib：`sz000001` / `sh600000` / `bj430047`

---

## 四、使用方法

### 1. 小样本验证（推荐先做）

```powershell
python run_all.py --limit 3 --start 2026-09-01
```

### 2. 全量构建

```powershell
# 修改 config.py 中 START_DATE 后：
python run_all.py
# 或指定起点：
python run_all.py --start 2020-01-01
```

### 3. 分步执行

```powershell
python scripts/01_universe.py
python scripts/02_fetch_minute.py
python scripts/03_fetch_adjfactor.py
python scripts/04_build_and_dump.py
```

### 4. 数据健康检查

```powershell
python verify_read.py
# 结果见 logs/verify_read_out.txt
```

### 5. 日常增量更新

收盘后重跑即可（分钟数据与复权因子均为增量追加，历史不复权价与定点复权因子不会改变）：

```powershell
python run_all.py
```

---

## 五、数据口径与注意事项

1. **复权方式**：采用「不复权价 × Wind `adjfactor` = 后复权价」，与 Qlib 官方数据口径一致；`volume` 保持原始成交量，`factor` 字段存复权因子。
   - 已核验：`11.31 × 87.209085 = 986.33`（平安银行 2026-10-28 后复权收盘）。
2. **收盘集合竞价**：A 股 14:57–15:00 为收盘集合竞价，Wind 仅在 `15:00` 给出一根 bar，`14:58/14:59` 为 OHLC 全空的占位行，已在拉取与组装时剔除。
3. **NaN 含义**：对齐到统一分钟日历后，个股停牌或原始 bar 偶发缺失（如个别日缺少 11:30 bar）会产生 NaN，属正常现象；因子计算时建议用 `ffill` 等填充。
4. **限频与重试**：`config.py` 中 `WIND_SLEEP`、`WIND_RETRY` 控制请求间隔与重试；失败的股票重跑即可续传，不会重复拉取。
5. **数据量与耗时**：全 A 约 5577 只，单只每天 240 根 bar；分钟数据量大，首次全量构建耗时较长，建议先小样本验证、再分批（可用 `--limit` 或调整股票池）后台运行。
6. **幸存者偏差**：当前股票池为指定日期的**在市全 A**，不含已退市股票；若需严格避免幸存者偏差，需额外拉取历史成分与退市股票（可后续扩展）。
7. **分钟回溯长度**：Wind 分钟数据的可回溯区间取决于账号权限，2010 年之后通常较完整。

---

## 六、后续可扩展

- 因子库：`py-alpha-lib`（Alpha101/191）、`qweave`（450+ 因子）。
- 自动因子挖掘：`AlphaGen`（强化学习，KDD 2023）、`RD-Agent`（LLM 多智能体）。
- 因子评估：`alphalens-reloaded`、`jqfactor_analyzer`。
- 模型与回测：Qlib 内置 LightGBM / LSTM / Transformer 等，`qrun` 一键跑通因子 → 模型 → TopkDropout → 回测。
