# 03 · KNN（K-Nearest Neighbors）

> K 近邻——最直观的"懒惰学习"算法。看你的邻居长啥样，就猜你长啥样。

---

## 学什么用什么

| 目标 | 入口 |
|---|---|
| 系统铺概念 | [`00-KNN算法-知识点大纲.md`](./00-KNN算法-知识点大纲.md) |
| 上课节奏 / active recall 表 | [`01-LESSON-PLAN.md`](./01-LESSON-PLAN.md) |
| 知识图（一图流） | [`02-KNOWLEDGE-MAP.txt`](./02-KNOWLEDGE-MAP.txt) |
| 代码 + 复习题（按主题分组） | [`lab/`](./lab/) |
| 学完沉淀（答案区） | [`99-DISTILL.md`](./99-DISTILL.md) |

---

## lab 主题模块（`lab/<主题>/`）

每个主题目录下放配对的 `.py` 代码 + `.md` 复习题。

| 主题 | 代码 | 复习题 | 答案 | 覆盖知识点 |
|---|---|---|---|---|
| `00-basicapi` | [`step0_api_classification.py`](./lab/00-basicapi/step0_api_classification.py) | [`step0-api-classification.md`](./lab/00-basicapi/step0-api-classification.md)（9 题） | [`step0-api-classification-answers.md`](./lab/00-basicapi/step0-api-classification-answers.md) | sklearn 三件套：分类 + 回归 API 最小例 |
| `01-bruteforce` | [`step1_brute_force_timing.py`](./lab/01-bruteforce/step1_brute_force_timing.py) | [`step1-brute-force-timing.md`](./lab/01-bruteforce/step1-brute-force-timing.md)（10 题） | — | brute-force 邻居查找耗时基准（fit 几乎免费 / predict 是大头） |
| `02-scaling` | [`step2_feature_scaling.py`](./lab/02-scaling/step2_feature_scaling.py) | [`step2-feature-scaling.md`](./lab/02-scaling/step2-feature-scaling.md)（11 题） | [`step2-feature-scaling-answers.md`](./lab/02-scaling/step2-feature-scaling-answers.md) | MinMaxScaler / StandardScaler：KNN 必备前置步骤 |
| `03-iris-explore` | [`step3_iris_explore.py`](./lab/03-iris-explore/step3_iris_explore.py) | [`step3-iris-explore.md`](./lab/03-iris-explore/step3-iris-explore.md)（10 题） | [`step3-iris-explore-answers.md`](./lab/03-iris-explore/step3-iris-explore-answers.md) | sklearn 数据集 API / numpy↔pandas / seaborn 可视化 |
| `04-split-stratify` | [`step4_split_stratify.py`](./lab/04-split-stratify/step4_split_stratify.py) | [`step4-split-stratify.md`](./lab/04-split-stratify/step4-split-stratify.md)（10 题） | [`step4-split-stratify-answers.md`](./lab/04-split-stratify/step4-split-stratify-answers.md) | train_test_split / random_state / stratify 分层抽样 |
| `05-iris-pipeline` | [`step5_iris_pipeline.py`](./lab/05-iris-pipeline/step5_iris_pipeline.py) | [`step5-iris-pipeline.md`](./lab/05-iris-pipeline/step5-iris-pipeline.md)（11 题） | [`step5-iris-pipeline-answers.md`](./lab/05-iris-pipeline/step5-iris-pipeline-answers.md) | 监督学习 6 步标准流程 / model.score / predict_proba |
| `06-cv-gridsearch` 🔥 | 朴素版 [`step6_cv_gridsearch_naive.py`](./lab/06-cv-gridsearch/step6_cv_gridsearch_naive.py) <br> 正确版 [`step6_cv_gridsearch_pipeline.py`](./lab/06-cv-gridsearch/step6_cv_gridsearch_pipeline.py) | [`step6-cv-gridsearch.md`](./lab/06-cv-gridsearch/step6-cv-gridsearch.md)（16 题） | [`step6-cv-gridsearch-answers.md`](./lab/06-cv-gridsearch/step6-cv-gridsearch-answers.md) | **CV / GridSearch / Pipeline / data leakage 对比**（金矿） |
| `07-mnist` | 探索 [`step7_mnist_explore.py`](./lab/07-mnist/step7_mnist_explore.py) <br> 训练 [`step7_mnist_train.py`](./lab/07-mnist/step7_mnist_train.py) <br> 推理 [`step7_mnist_predict.py`](./lab/07-mnist/step7_mnist_predict.py) | [`step7-mnist.md`](./lab/07-mnist/step7-mnist.md)（20 题） | [`step7-mnist-answers.md`](./lab/07-mnist/step7-mnist-answers.md) | MNIST 端到端 / x/255 捷径 / joblib / 200MB 模型真相 |

---

## 上下文锚

- 算法定位：lazy learner，fit 不训练，predict 才算。和逻辑回归/线性回归（eager）反着来。
- 工程类比：`fit ≈ 把训练集塞进数据库 / predict ≈ 每次查询都遍历表算距离`
- 必踩的坑：忘了标准化（量纲大的特征霸占距离）、k 选偶数（平票）、高维（维度灾难）、生产部署（要打包整个训练集）
