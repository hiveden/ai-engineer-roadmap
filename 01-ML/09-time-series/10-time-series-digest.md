# L1 备课摘要 · 时间序列预测 / ARIMA

> 目标：为第 10 节「时间序列 / ARIMA」做 L1 级备课储备。课上讲的是经典 ARIMA 与滑窗特征工程，但 Alex 问出来的问题大概率是「2025 年还用 ARIMA 吗？TimeGPT 这类基础模型怎么定位？和 AI Agent 的监控告警怎么接？」—— 下面按七个维度准备下限答案。

---

## 1. 候选数据集清单

| 数据集 | 规模 / 频率 | 适用教学点 | 备注 |
|---|---|---|---|
| **南方电网电力负荷**（原始培训笔记场景） | 小时级，单变量 + 强日/周/年周期 | 滑窗特征工程、XGBoost 拍扁宽表、业务锚（工作日 vs 周末） | Alex 已在 [`08-Clustering-and-Time-Series-Project.md`](../../01-Machine-Learning-Foundation/08-Clustering-and-Time-Series-Project.md) 见过一版 |
| **airline-passengers** | 月度，144 条，经典 | ARIMA(p,d,q) 入门、ADF 检验、季节性分解 STL | Box-Jenkins 教科书数据集，statsmodels 自带（[jbrownlee/Datasets](https://github.com/jbrownlee/Datasets/blob/master/airline-passengers.names) / [Rdatasets mirror](https://vincentarelbundock.github.io/Rdatasets/doc/datasets/AirPassengers.html)） |
| **比特币 / 股价**（yfinance） | 日/分钟级 | 讲「为什么金融时序 ARIMA 很快翻车」—— 非平稳、黑天鹅、效率市场假说 | 反面教材锚点（[ranaroussi/yfinance](https://github.com/ranaroussi/yfinance) / [官方文档](https://ranaroussi.github.io/yfinance/)） |
| **零售销量**（M5 Walmart、Rossmann） | 日级，多店铺多商品 | Prophet 假日效应、层级预测 | 业务感最强，接 Alex 的全栈背景（[M5 Forecasting - Accuracy · Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy)） |
| **服务器监控指标**（CPU / QPS / 延迟，Prometheus 导出） | 秒/分钟级 | 对接 AIOps、Agent 可观测性 | **Alex 目标领域直接相关**（[prometheus/prometheus](https://github.com/prometheus/prometheus) / [AICoE/prometheus-data-science](https://github.com/AICoE/prometheus-data-science)） |

---

## 2. 算法现代实践 · 时效性素材

### 2024-2025 态势：三层格局

**第一层 · 经典统计（依然活着）**
ARIMA / SARIMA / ETS 在 2025 年并没有退场。论文实测：简单线性模式下 ARIMA 的 MAPE 稳定在 3.2–13.6%，且**可解释、训练秒级、不需要 GPU**。适用场景收敛到：短序列（< 几百点）、单变量、需要给审计解释「为什么这么预测」。

**第二层 · Prophet（Meta 2017 开源，工程师最爱）**
Prophet 把时序拆成 `trend + seasonality + holidays + noise`，参数语义化（`changepoint_prior_scale`、`seasonality_mode`），业务人员都能调。对带强季节性 + 假日效应的业务时序 MAPE 能做到 2.2–24.2%。**定位：ARIMA 的「工程化糖衣」**——不懂 p/d/q 的工程师也能出像样结果，代价是长期非线性依赖建模能力弱。

**第三层 · 深度 + 基础模型（2024-2025 爆发）**

| 模型 | 出品方 | 关键特性 | 发布时间 |
|---|---|---|---|
| **TimeGPT** | Nixtla | 闭源 API，100B+ 数据点训练，零样本（[Nixtla/nixtla](https://github.com/Nixtla/nixtla) / [arxiv 2310.03589](https://arxiv.org/abs/2310.03589)） | 2023-2024 |
| **Chronos** | Amazon | 开源，基于 T5 tokenize 时序，Chronos-2 加多变量（[amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting) / [Chronos-2 博客](https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting)） | 2024 |
| **TimesFM** | Google | 开源 decoder-only，patch-based，点预测（[google-research/timesfm](https://github.com/google-research/timesfm) / [Google Research 博客](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/)） | 2024 |
| **Moirai** | Salesforce | 开源，MoE Transformer，多变量原生，LOTSA 27B 数据（[SalesforceAIResearch/uni2ts](https://github.com/SalesforceAIResearch/uni2ts) / [Salesforce 博客](https://www.salesforce.com/blog/moirai/) / [Moirai-MoE arxiv 2410.10469](https://arxiv.org/abs/2410.10469)） | 2024，Moirai 2.0 于 2025 |
| **Toto** | Datadog | 面向可观测性数据专门训练的 TSFM（[DataDog/toto](https://github.com/DataDog/toto) / [Datadog 发布博客](https://www.datadoghq.com/blog/datadog-time-series-foundation-model/) / [arxiv 2407.07874](https://arxiv.org/abs/2407.07874)） | 2024-2025 |

核心意义：像 NLP 里 BERT/GPT 一样，时序领域也出现「先预训练，后零样本或微调」的范式。Datadog 的 Toto 直接用于监控指标，**这条线和 Alex 的 Agent 可观测性方向强相关**。

### 什么时候仍然用 ARIMA（Tech Lead 视角）

- 数据点 < 500，深度模型必过拟合
- 需要置信区间且老板/审计要一眼看懂参数含义
- 只跑单台机器、无 GPU，且要分分钟出结果
- 单变量 + 强平稳性 + 业务变化慢（能源负荷短期、库存管理）

---

## 3. 常见学员追问 · 下限答案

1. **时序和普通回归区别？**
   普通回归假设样本 i.i.d.（独立同分布），时序样本**有自相关**（今天和昨天强相关），随机打乱会毁掉时间结构。评估必须按时间切分。参见 [Hyndman & Athanasopoulos · FPP3](https://otexts.com/fpp3/)。

2. **平稳性（stationarity）是什么？ADF 检验？**
   平稳 = 均值、方差、自协方差不随时间变。ARIMA 假设序列平稳。**ADF 检验（Augmented Dickey-Fuller）**：零假设「序列非平稳」，p < 0.05 拒绝 → 序列平稳。业务锚：「数据的统计性质稳定」= 昨天的规律今天还能用。参考 [statsmodels adfuller 文档](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html) / [Wikipedia: Augmented Dickey-Fuller test](https://en.wikipedia.org/wiki/Augmented_Dickey%E2%80%93Fuller_test) / [Stationarity and detrending (ADF/KPSS) · statsmodels](https://www.statsmodels.org/dev/examples/notebooks/generated/stationarity_detrending_adf_kpss.html)。

3. **AR / MA / ARIMA / SARIMA 分别是什么？**
   - AR(p)：今天 = 过去 p 天的加权和 + 噪声
   - MA(q)：今天 = 过去 q 个噪声的加权和 + 均值
   - ARIMA(p,d,q)：做 d 阶差分后，套 AR(p) + MA(q)（[statsmodels ARIMA 文档](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)）
   - SARIMA(p,d,q)(P,D,Q)[s]：再叠一层季节性，s 是季节周期（月度数据 s=12）（[statsmodels SARIMAX 文档](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html) / [FPP3 §9.9 Seasonal ARIMA](https://otexts.com/fpp3/seasonal-arima.html)）

4. **d 阶差分什么意思？**
   差分 = 相邻项相减（`y_t - y_{t-1}`），用来**消除趋势 / 季节性**让序列变平稳。d=1 一次差分（消线性趋势），d=2 二次差分（消二次趋势）。业务锚：看「增量」而非「存量」，像看每日 DAU 增长数而非总 DAU。参见 [FPP3 §9.1 Stationarity and differencing](https://otexts.com/fpp3/stationarity.html)。

5. **如何选 p / d / q？**
   看 **ACF（自相关图）和 PACF（偏自相关图）**：PACF 截尾 q 步 → AR(p=q)；ACF 截尾 q 步 → MA(q)。现代工程派直接用 `pmdarima.auto_arima` 暴力搜索 AIC 最小组合，跟自动调参一回事。参见 [pmdarima.auto_arima 文档](https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html) / [Duke · Identifying the orders of AR and MA terms](https://people.duke.edu/~rnau/411arim3.htm)。

6. **Prophet 为什么比 ARIMA 好用？**
   工程化封装：自动假日、自动 changepoint、缺失值鲁棒、参数语义化。代价：长程非线性依赖弱于 LSTM/Transformer。适用：业务时序（电商 GMV、广告曝光、门店销量）。参考 [Prophet 官方文档](https://facebook.github.io/prophet/) / [Taylor & Letham · Forecasting at Scale (2017)](https://facebook.github.io/prophet/static/prophet_paper_20170113.pdf) / [FPP3 §12.2 Prophet model](https://otexts.com/fpp3/prophet.html)。

7. **LSTM / Transformer-based 时序模型何时值得？**
   - 有足够数据（至少几千到几万点）
   - 多变量 + 长程依赖 + 非线性
   - 长 horizon 预测（Informer、Autoformer、PatchTST 在这里吊打 LSTM）—— 参见 [Informer · arxiv 2012.07436](https://arxiv.org/abs/2012.07436)（AAAI 2021 Best Paper）/ [PatchTST · arxiv 2211.14730](https://arxiv.org/abs/2211.14730)（ICLR 2023，[GitHub](https://github.com/yuqinie98/PatchTST)）
   - 2025 实践：直接考虑基础模型（Chronos、TimesFM）零样本或少量微调，而非从零训 LSTM

8. **为什么 `train_test_split` 不能随机切？**
   随机打乱会**把未来样本混进训练集**——相当于用下周的数据训练出"能预测今天"的模型，典型**数据泄漏（data leakage）**。正确做法：按时间切分（前 80% 训练，后 20% 测试），或用 `TimeSeriesSplit` 做滚动交叉验证。参考 [sklearn TimeSeriesSplit 文档](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) / [sklearn §3.1 Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split) / [Hyndman · Cross-validation for time series](https://robjhyndman.com/hyndsight/tscv/)。

9. **时序预测和 AI Agent 什么关系？**（Alex 目标域）
   三个主线：
   - **可观测性预测**：Agent 调用 LLM 的 latency / token 消耗 / 错误率曲线 → 预测异常和容量瓶颈，典型 Datadog Toto 场景（[Toto & BOOM 博客](https://www.datadoghq.com/blog/ai/toto-boom-unleashed/) / [Toto arxiv 2407.07874](https://arxiv.org/abs/2407.07874)）
   - **异常检测驱动 Agent 动作**：时序预测置信区间被打破 → 触发 Agent 自动排障（agentic AIOps）
   - **成本预测 / 容量规划**：AI Agent 产品按 token 计费，需要预测下周 API 成本、GPU 占用
   行业信号：Datadog 2025 Q2 被 Forrester 评为 AIOps Leader，预警能提前 20-40 分钟。

---

## 4. 业界翻车 / 反面教材

1. **随机切分 = 数据泄漏**：用 `train_test_split(shuffle=True)` 做时序验证，指标好看但线上崩。南方电网笔记里的 `random_state=22` 就是典型反面——是原始材料的 bug，要在讲课时点出来。参考 [MachineLearningMastery · Data Leakage in ML](https://machinelearningmastery.com/data-leakage-machine-learning/) / [sklearn Common Pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)。
2. **忽视季节性**：只做 ARIMA(p,d,q) 不做 SARIMA，在有日/周/年周期的数据上残差严重。参考 [FPP3 §9.9 Seasonal ARIMA](https://otexts.com/fpp3/seasonal-arima.html) / [Duke · General seasonal ARIMA models](https://people.duke.edu/~rnau/seasarim.htm)。
3. **过度依赖 ARIMA 处理长序列**：10 万点 + 多变量还硬上 ARIMA，训练爆内存，精度输给 Prophet 一大截。参考 [Hyndman · Forecasting with long seasonal periods](https://robjhyndman.com/hyndsight/longseasonality/) / [neptune.ai · ARIMA vs Prophet vs LSTM](https://neptune.ai/blog/arima-vs-prophet-vs-lstm)。
4. **LSTM 过拟合短数据**：几百条样本硬上 3 层 LSTM，训练 loss 完美、测试集随机猜。参考 [MachineLearningMastery · Diagnose Overfitting and Underfitting of LSTM Models](https://machinelearningmastery.com/diagnose-overfitting-underfitting-lstm-models/)。
5. **基础模型（TimeGPT 等）盲目信仰**：零样本方便，但专业领域（电力负荷、金融高频）未必赢过精调 XGBoost + 好特征工程。参考 [Grid Dynamics · TSFM Comparison](https://www.griddynamics.com/blog/ai-models-demand-forecasting-tsfm-comparison) / [Observability Perspective on TSFMs · arxiv 2505.14766](https://arxiv.org/abs/2505.14766)。
6. **不做 backtesting**：只看一次 hold-out 测试，忽略**滚动预测（walk-forward validation）**；真实业务中每个时段稳定性才是关键。参考 [Hyndman · Cross-validation for time series](https://robjhyndman.com/hyndsight/tscv/) / [MachineLearningMastery · How To Backtest ML Models for Time Series](https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/)。

---

## 5. 推荐阅读清单

- **[statsmodels ARIMA 文档](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)** —— 官方 API，Python 工程师必备
- **[Prophet 官方文档](https://facebook.github.io/prophet/)** —— Meta 出品，quickstart 能跑通
- **[Forecasting: Principles and Practice (Hyndman, 3rd ed. 在线免费)](https://otexts.com/fpp3/)** 与 Pythonic 版 [fpppy](https://otexts.com/fpppy/) —— 时序领域圣经，第 15 章专讲基础模型
- **[Chronos 论文 / 博客](https://towardsdatascience.com/chronos-the-rise-of-foundation-models-for-time-series-forecasting-aaeba62d9da3/)** —— Amazon 开源 TSFM 入门
- **[TimesFM 解读](https://towardsdatascience.com/timesfm-the-boom-of-foundation-models-in-time-series-forecasting-29701e0b20b5/)** —— Google 的 decoder-only TSFM
- **[Datadog Toto 博客](https://www.datadoghq.com/blog/datadog-time-series-foundation-model/)** —— 面向可观测性的 TSFM，Alex 的目标领域锚点
- **[ARIMA vs Prophet vs LSTM 横评（neptune.ai）](https://neptune.ai/blog/arima-vs-prophet-vs-lstm)** —— 决策树图最清晰的一篇
- **[TSFM 对比 · Grid Dynamics](https://www.griddynamics.com/blog/ai-models-demand-forecasting-tsfm-comparison)** —— 四大基础模型横评
- **[AIOps 时序异常检测 Survey (arxiv 2308.00393)](https://arxiv.org/abs/2308.00393)** —— 接 Agent 可观测性方向

---

## 6. 盲区填补要点 · 对接 AI Agent 监控/运维方向

这是 Alex 真正要的，单拎一段。

**场景 1 · Agent 服务 SLO 预测**
多 Agent 系统里，每个 Agent 调用 LLM 都有 latency 曲线。用时序预测（Prophet 或 Toto）建模正常 baseline，置信区间被打破 → 触发告警 → 上层 Agent 自动降级（切小模型、切本地 Ollama）。**这是 ML 能直接嵌入 Agent 架构的点**，不需要 DL 炼丹。

**场景 2 · Token 成本 / 容量预测**
Claude Max 订阅 / API 账单有明显的日/周季节性（工作日 peak、周末低谷）。用 SARIMA 或 Prophet 预测下周消耗，超出预算自动切换到 Ollama 本地模型。**工程落地只需 statsmodels + 定时任务**，不需要 GPU。

**场景 3 · 多 Agent 协作中的时序特征**
比如 "监控 Agent" 的输入是 Prometheus metric 流 → 特征工程（滑窗、差分）→ 喂给 XGBoost / TSFM → 输出风险分数 → "决策 Agent" 基于风险分数做路由。这正是原始培训笔记里电力负荷架构的升级版（Python 炼丹 → ONNX → Go 推理，0.5ms 单次）。

**硬技能投入建议（L1 阶段）**
- 必掌握：statsmodels ARIMA + Prophet 两个库能跑通
- 了解即可：基础模型（Chronos/TimesFM）的 API 调用方式，不要求理解 Transformer 内部
- 可以跳过：从零训练 LSTM —— 2025 年这条路线 ROI 很低，基础模型接过去了

---

## 7. 与模板 §8.9 三源定位映射

| 维度 | 原始培训笔记 | 第一版改编（08 号文档） | 本摘要新增 |
|---|---|---|---|
| **算法定位** | 时序预测算法清单（朴素→AR→MA→ARIMA→DL），以 ARIMA 为「经典但准确性不如 ML」 | 聚焦滑窗特征工程拍扁宽表 + XGBoost，把 ARIMA 当历史背景 | 2024-2025 TSFM 爆发（Chronos/TimesFM/Moirai/Toto），补「何时仍用 ARIMA」的现代判据 |
| **业务锚** | 南方电网电力负荷单一场景 | 电网 + 冷启动类比 | 扩展到 AIOps / Agent 可观测性 / Token 成本，贴合 Alex 目标域 |
| **工程落地** | Python 单进程 XGBoost，train.py + predict.py | Python 训练 → ONNX → Go 推理 0.5ms 的跨语言架构 | 补 Redis 滑窗缓存细节 + 讲清"为什么不能用 Flask 在线 serve Python" |
| **反面教材** | 无显式提醒（代码里 `random_state=22` 随机切分是 bug） | 指出"XGBoost 没记忆、要拍扁" | 显式列出 6 条翻车点，第 1 条就是原始笔记自己的 bug |
| **盲区** | ADF 检验、ACF/PACF、Prophet、DL 几乎没讲 | RNN/Transformer 作为「深度学习伏笔」一笔带过 | 补 ADF / ACF/PACF 下限答案、Prophet 定位、TSFM 对比表 |
| **可解释性** | 未涉及 | 未涉及 | 补「为什么 ARIMA 在审计场景不可替代」 |
| **评估方法** | 只讲 MAE/MSE | 同 | 补 walk-forward validation、时间切分、为什么不能 shuffle |

**三源定位结论**：原始笔记偏"统计算法 + 特征工程"的教科书脉络；第一版改编补了工程落地架构；本摘要在此基础上补 2025 现代判据 + Agent 方向盲区，形成 L1 备课完整闭环。课堂主线建议仍以南方电网 ARIMA/XGBoost demo 作锚，但在 Q&A 环节主动引出"TimeGPT / Toto / Agent 可观测性"以贴合 Alex 目标。
