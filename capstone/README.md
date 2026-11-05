# 多因子选股量化策略 · 端到端作品集（Capstone）

一个自包含、可复现的 A 股量化研究项目，完整覆盖从原始数据到可交易组合的全链路，
并通过严格的防未来函数与成本建模保证结果可信。面向多因子选股量化研究员岗位，
兼顾 CTA / 高频 / 衍生品 / 量化开发的知识迁移。

## 一、研究链路

```
数据加载        Qlib 后复权 OHLCV + 市值（沪深300，300只，2018 至今）
   ↓
因子构建        5日反转 / 20日波动 / 市值 / 60日动量（截面标准化）
   ↓
因子评估        截面 RankIC、IR、IC 胜率
   ↓
标签与特征      未来5日收益截面 rank 标签（比原始收益稳健）
   ↓
模型预测        LightGBM 年度滚动 walk-forward，杜绝未来函数
   ↓
组合构建        月末按预测分选 Top30 等权，权重行和恒为 1
   ↓
回测引擎        weights.shift(1) 防未来函数，双边 20bps 成本
   ↓
绩效与风控      年化 / 夏普 / 最大回撤 / 净值与回撤图
   ↓
模拟盘衔接      接入 live_lab2 实时行情→风控→自适应执行→报警接管
```

## 二、核心结果（真实数据，含交易成本）

| 指标 | 策略 | 基准（沪深300等权） |
|---|---|---|
| 年化收益 | 16.00% | 14.85% |
| 夏普比率 | 1.01 | 0.80 |
| 最大回撤 | -24.01% | -26.12% |
| 样本外 RankIC | 0.0143（胜率 53.6%） | — |
| 年化换手成本 | 2.44% | — |

策略在收益、夏普、回撤三个维度均优于基准，且样本外预测保持正 IC。

## 三、目录结构

```
capstone/
├── capstone_strategy.ipynb   端到端可运行 notebook（主交付）
├── capstone_strategy.html    notebook 渲染版（免环境查看）
├── README.md                 本文件：项目说明与复现指南
├── 面试题库.md               分模块高频面试题与参考要点
├── 简历项目陈述.md           简历 bullet 与项目口述话术
└── 模拟面试记录.md           模拟压力面问答与追问拆解
```

## 四、环境与复现

- Python 3.10，conda 环境 `qlib`；核心依赖：qlib 0.9.7、pandas、numpy、lightgbm、scipy、matplotlib。
- 数据：本地 Qlib 日频 bin（`data/qlib_daily`），自动读到最新交易日，无需写死日期。

复现步骤：

```powershell
# 1. 激活环境
conda activate qlib
# 2. 执行 notebook（指定 qlib kernel）
jupyter nbconvert --to notebook --execute --inplace `
  capstone\capstone_strategy.ipynb --ExecutePreprocessor.kernel_name=qlib
# 3. 或直接在 Jupyter 中打开 capstone\capstone_strategy.ipynb 逐格运行
```

固定随机种子 `SEED=42`，结果可复现。

## 五、工程规范与研究亮点

- **防未来函数**：标签用未来收益、特征只用历史；回测用 `weights.shift(1)`；模型 walk-forward 滚动训练。
- **真实成本建模**：按换手计算双边 20bps 交易成本，避免零成本虚高。
- **数据质量处理**：识别恢复上市首日不设涨跌幅的极端跳变（如盐湖股份 2021-08-10 +306%），
  并在组合构建中保证权重行和恒为 1，避免杠杆累积。
- **参数化股票池**：通过 `l2.config.load_universe` 一套逻辑切换沪深300 / 中证500 / 全A。

## 六、可改进方向

1. 扩充因子库（量价、基本面、另类数据）与股票池（中证500、全A）；
2. 组合层加入行业中性、风格暴露约束与优化器（cvxpy），替代等权；
3. 更精细的 TCA、容量分析与模型集成、在线监控（PSI/IC 衰减）；
4. 接入实盘模拟盘并持续跟踪，验证回测与实盘一致性。
