# 05 - 决策树与规则抽取：可解释性是金融风控的硬通货

在所有传统机器学习算法里，决策树（Decision Tree）是**最像"传统代码"**的那一个——它的本质就是一棵可序列化的 `if-else` 嵌套树。

对于一个写了 7 年 Java/Go 的全栈工程师来说，理解决策树的成本几乎为零：
- 一棵决策树 ≈ **一个由数据"自动写出来"的 Drools 规则引擎**
- 模型训练 ≈ **从历史样本中归纳出 if-else 链路**
- 模型预测 ≈ **沿着 if-else 链路走一次，O(树深度)**

这是你跨入机器学习世界时遇到的**第一个可以直接读懂、直接审计、直接拿给法务和监管看的模型**。在金融风控这种"模型必须解释清楚"的领域，它的地位至今无人撼动。

---

## 1. 降维认知：决策树是"自己写自己"的规则引擎

### 1.1 一个开发者秒懂的例子

相亲场景，候选人是 26 岁、长相一般、收入中等、税务局公务员，决策过程：

```java
if (age > 30) return REJECT;
else if (looks == "丑") return REJECT;
else if (income == "低") {
    if (isCivilServant) return ACCEPT;
    else return REJECT;
} else return ACCEPT;
```

这就是一棵决策树。区别在于：**传统软件 1.0 时代，这套规则是产品经理拍脑袋写的；而决策树是从数据里自动学出来的。**

### 1.2 树结构的三要素

| 节点类型 | 数据结构对应物 | 作用 |
|---|---|---|
| **根节点 / 内部节点** | 一个 `switch` 或 `if` 判断 | 在某个特征上做条件分裂 |
| **分支** | `case` 或 `else if` 走向 | 判断结果的不同出口 |
| **叶节点** | 最终的 `return` | 输出预测类别（分类）或数值（回归） |

**训练 = 自动归纳出这棵 if-else 树的形状与判断条件。**
**预测 = 给定一条新数据，从根节点走到叶节点，O(log n)。**

---

## 2. 选树算法 ≈ 选数据库引擎：ID3 / C4.5 / CART 的工程取舍

历史上有三代主流的决策树算法。**你不需要记住它们的数学公式**——只需要像选 MySQL 引擎（InnoDB / MyISAM / TokuDB）一样，记住它们的工程定位：

| 算法 | 提出年份 | 分裂依据（一句话） | 工程短板 | 类比 |
|---|---|---|---|---|
| **ID3** | 1975 | "哪个特征能让数据最'变干净'，就选哪个" | 只能处理离散字段；偏爱取值多的字段（比如 user_id 这种字段会被无脑选中） | MyISAM——上古、有缺陷 |
| **C4.5** | 1993 | ID3 的改良版，惩罚了"取值过多"的字段 | 不能处理超大数据集，吃内存 | InnoDB——主流、稳定 |
| **CART** | 1984 | 用更轻量的指标，一律生成**二叉树**；既能分类又能回归 | 二叉强制约束，树更深 | RocksDB——工程化最强 |

### 2.1 真正要记住的工程结论

- **`scikit-learn` 默认实现的是 CART**。`sklearn.tree.DecisionTreeClassifier` 和 `DecisionTreeRegressor` 用的都是 CART。
- **CART 是当代 XGBoost / LightGBM 这些集成模型的基础构件**。理解了 CART，下一章集成学习就是顺水推舟。
- **特征选择的本质**：每次分裂都问一个问题——"如果按这个字段切一刀，左右两堆数据会不会更'纯'？" 越纯越好。**怎么量化"纯"，就是 ID3/C4.5/CART 的分歧点**。剩下的数学推导对工程师没有价值。

> 💡 **降维提示**：当你看到"信息熵""信息增益""基尼指数"这些术语时，把它们统一理解为**"不纯度的不同量化方法"**，就够了。它们就像 MD5/SHA1/SHA256——都是哈希函数，区别只在于实现细节。

---

## 3. 致命陷阱：过拟合与剪枝（Pruning）

### 3.1 为什么决策树天生过拟合

如果不加约束，决策树会一直分裂到每个叶节点只剩 1 条样本——也就是**死记硬背了整个训练集**。这种树在训练集上准确率 100%，上线即灾难。

类比：**就像一个把所有 SQL 查询结果都缓存到 Redis 里、且永不过期的系统**——查询飞快，但只要数据稍有变动，缓存全部失效。

### 3.2 剪枝：两种工程策略

| 策略 | 时机 | 思路 | 工程类比 |
|---|---|---|---|
| **预剪枝（Pre-pruning）** | 构建过程中 | 边长边判断："这一刀切下去验证集精度反而下降，那就别切" | **熔断器**：检测到下游异常立即停止扩张 |
| **后剪枝（Post-pruning）** | 构建完成后 | 先长成完整大树，然后从底向上裁："这个子树砍掉换成叶节点是不是更准？" | **GC**：先放任分配，再回头扫一遍清理掉无用对象 |

### 3.3 sklearn 的工程化超参数

`DecisionTreeClassifier` 暴露给你的几个关键"防过拟合"旋钮，本质都是预剪枝：

```python
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(
    criterion='gini',          # 用基尼指数（CART 默认）
    max_depth=8,               # 树最深 8 层 —— 最重要的旋钮
    min_samples_split=20,      # 节点至少 20 个样本才允许继续分裂
    min_samples_leaf=10,       # 叶节点至少 10 个样本
    random_state=42
)
```

**架构师视角**：这几个参数本质上是在说"**别给我把规则写得太细**"。`max_depth` 越小，树越浅，泛化越好但欠拟合；越深越容易死记硬背。**实战中 `max_depth` 在 5~10 之间是常见甜区**。

---

## 4. 工程优势：可解释性 = 监管的硬通货

在金融、医疗、保险这些强监管行业，**模型必须能向监管机构和用户解释清楚"为什么拒绝你的贷款"**——这是法律红线，不是技术选型问题。

### 4.1 决策树 vs 黑盒模型

| 模型 | 解释能力 | 监管友好度 | 典型场景 |
|---|---|---|---|
| 决策树 | ⭐⭐⭐⭐⭐ 每条规则都能拉出来给人看 | 极高 | **银行反欺诈、信贷评分卡、医疗辅助诊断** |
| 神经网络 / LLM | ⭐ 几乎是黑盒 | 极低（除非加 SHAP/LIME 解释器） | 推荐系统、图像识别 |

### 4.2 规则抽取：把树"反编译"成 SQL/代码

`scikit-learn` 训练出的决策树可以直接导出为人类可读的 if-else 文本，甚至可以反向生成 SQL：

```python
from sklearn.tree import export_text

rules = export_text(clf, feature_names=['年龄', '收入', '负债比', '逾期次数'])
print(rules)
# |--- 逾期次数 <= 2.50
# |   |--- 收入 <= 8000
# |   |   |--- class: 拒绝
# |   |--- 收入 >  8000
# |   |   |--- class: 通过
# |--- 逾期次数 >  2.50
# |   |--- class: 拒绝
```

**这段输出可以直接交给法务、合规、业务方审核**，甚至可以**手动改两条规则后再翻译回 SQL 上线**——这是任何深度学习模型都做不到的事。

---

## 5. 跨语言落地：决策树的两条部署路径

### 5.1 路径 A：PMML —— 给老牌金融系统的"普通话"

PMML（Predictive Model Markup Language）是 1998 年起就有的 XML 格式，**金融行业的事实标准**。各大银行的风控引擎几乎都内置了 PMML 解析器。

```python
# Python 端导出
from sklearn2pmml import sklearn2pmml, PMMLPipeline

pipeline = PMMLPipeline([("classifier", clf)])
sklearn2pmml(pipeline, "credit_scorer.pmml", with_repr=True)
```

```java
// Java 端加载（jpmml-evaluator 库）
Evaluator evaluator = new LoadingModelEvaluatorBuilder()
    .load(new File("credit_scorer.pmml"))
    .build();

Map<FieldName, FieldValue> input = new HashMap<>();
input.put(FieldName.create("年龄"), evaluator.encode(35));
input.put(FieldName.create("收入"), evaluator.encode(12000));
// ...
Map<FieldName, ?> result = evaluator.evaluate(input);
// 推理耗时：亚毫秒级，纯本地 JVM 计算
```

**PMML 的不可替代价值**：它不仅序列化了模型本身，**还序列化了完整的特征预处理 pipeline**（标准化、独热编码、缺失值填充）——彻底解决了"线下训练用 Python，线上推理用 Java，结果对不上"的状态漂移噩梦。

### 5.2 路径 B：ONNX —— 给现代云原生系统的"通用语"

如果你的栈是 Go / Node.js / Rust，PMML 生态有限，更现代的选择是 ONNX：

```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

onnx_model = convert_sklearn(
    clf,
    initial_types=[("input", FloatTensorType([None, 4]))]
)
with open("credit_scorer.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

```go
// Go 端用 onnxruntime-go 加载推理（伪代码）
session, _ := ort.NewSession("credit_scorer.onnx")
output, _ := session.Run(map[string]interface{}{
    "input": []float32{35, 12000, 0.4, 1},
})
// P99 延迟：1-2 毫秒
```

### 5.3 选型决策表

| 场景 | 推荐 |
|---|---|
| **银行核心系统、保险、证券**（Java 老栈，监管要求 PMML 审计） | **PMML** |
| **互联网公司、云原生、Go/Node 网关** | **ONNX** |
| **微服务网关需要超低延迟（< 1ms）** | 把树直接编译成 Go/Java 源码（用 [m2cgen](https://github.com/BayesWitnesses/m2cgen) 这种工具），跳过任何运行时 |

> 💡 **第三条路** `m2cgen` 可以把决策树**直接生成原生 Java/Go/C 源代码**，连模型文件都不需要——这就是决策树的"可解释 + 简单"带来的极致工程优势：**它可以被反编译回普通代码**。

---

## 6. 风控实战的典型架构位置

一个典型的银行小额信贷实时审批系统：

```
用户提交贷款申请
       ↓
  [Java 业务网关]
       ↓
  特征实时聚合（MySQL 历史 + Redis 实时行为 + Kafka 黑名单）
       ↓
  [本地 PMML 评估器]  ← 决策树评分卡（深度 6，30 条规则）
       ↓
  评分 < 60 → 拒绝
  评分 60~80 → 转人工
  评分 > 80 → 自动通过
       ↓
  审计日志（记录每次预测命中的规则路径，监管要求保留 5 年）
```

**关键工程点**：
1. **审计日志记录命中的规则路径**——这是决策树独有的能力。换成神经网络，你只能存"输入特征 + 输出概率"，无法解释为什么。
2. **规则路径可以做 A/B 测试**——发现某条规则误杀率高，可以临时禁用单条 if-else，不需要重新训练整个模型。
3. **冷启动友好**——决策树训练快、推理快、文件小（一棵 max_depth=8 的树通常 < 50KB），适合作为新业务上线的第一版模型。

---

## 7. 单棵树的天花板：为什么必须升维到集成学习

决策树虽然可解释性极强，但有一个无法回避的硬伤：**单棵树的预测精度天花板很低**。

- 树太浅 → 欠拟合，规则太粗
- 树太深 → 过拟合，死记硬背训练集
- 怎么调 `max_depth` 都很难找到完美平衡点

**业界的工程解法**：与其纠结一棵完美的树，不如训练**一片森林**——让 100 棵决策树投票，或者让 100 棵树串行补偿误差。

这就是下一章的主题——**集成学习（Random Forest / XGBoost）**。它们的底层构件就是 CART 决策树，但通过"多数投票"或"残差补偿"的工程化组合，把单棵树的精度推到了表格数据上的 SOTA。

> 单棵决策树：可解释性的王者，精度的青铜。
> 集成学习：可解释性的青铜，精度的王者。
> 在你的业务里二选一——还是两者都要（XGBoost + SHAP 解释器）？

---

## 8. 架构师面试题（自检）

1. **业务方说"为什么这个用户被风控拒绝了"，决策树和神经网络分别该怎么回答？**
2. **PMML 和 ONNX 在决策树场景下，应该如何选择？给出三个判断维度。**
3. **`max_depth=3` 和 `max_depth=20` 的决策树，哪个更适合作为新业务的冷启动模型？为什么？**
4. **如果让你用 Java 手写一个决策树推理器（不依赖任何库），你会怎么设计数据结构？序列化格式选什么？**
5. **决策树相对于 LLM 的最大架构优势是什么？最大劣势是什么？**
