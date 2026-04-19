"""
Step 1 — 看数据长什么样
问自己：每一行是什么？每一列是什么？Price 列是什么？

sklearn 内置数据集（不用联网，装库就带了）：
──────────────────────────────────────────────
load_*  = 小数据集，打包在库里，离线可用
fetch_* = 较大数据集，首次用时从网上下载，之后缓存在本地

常用的：
  load_iris            — 鸢尾花分类（150 条，3 类花，入门分类经典）
  load_digits          — 手写数字识别（1797 张 8×8 图片，0-9 分类）
  load_wine            — 红酒分类（178 条，3 类酒）
  load_breast_cancer   — 乳腺癌诊断（569 条，良性/恶性二分类）
  load_diabetes        — 糖尿病进展预测（442 条，回归）
  fetch_california_housing — 加州房价预测（20640 条，回归）← 本 demo 用的
  fetch_20newsgroups   — 新闻文本分类（18000+ 篇，20 个主题）
  fetch_olivetti_faces — 人脸识别（400 张 64×64 图片，40 个人）

代码知识点：
──────────────────────────────────────────────
import pandas as pd
  — pandas 是 Python 的表格数据处理库，pd 是约定俗成的别名
  — 类比：Java 里的 ResultSet / Go 里的 []map[string]interface{}，但功能强得多

from sklearn.datasets import fetch_california_housing
  — 从 sklearn 的 datasets 子模块导入一个函数
  — sklearn (scikit-learn) 是 Python 最主流的传统 ML 库

raw = fetch_california_housing()
  — 调用函数，返回一个 Bunch 对象（类字典）
  — raw.data  → 二维数组 (20640, 8)，每行一个街区，每列一个特征
  — raw.target → 一维数组 (20640,)，每个街区的房价中位数（×10万美元）
  — raw.feature_names → 8 个特征的列名列表

df = pd.DataFrame(raw.data, columns=raw.feature_names)
  — 把 numpy 数组转成 DataFrame（带行号、列名的表格）
  — data=raw.data → 表格内容（数字矩阵）
  — columns=raw.feature_names → 给每列起名字

df["Price"] = raw.target
  — 给 DataFrame 加一列叫 "Price"，值是 raw.target
  — 类比 SQL：ALTER TABLE df ADD COLUMN Price ...

df.head()
  — 返回前 5 行，快速预览数据。head(10) 就是前 10 行

len(df)
  — 行数。len(df.columns) 是列数

列含义：
  MedInc     — 居民收入中位数（万美元）
  HouseAge   — 房屋年龄（年）
  AveRooms   — 平均房间数
  AveBedrms  — 平均卧室数
  Population — 区域人口
  AveOccup   — 平均每户住几人
  Latitude   — 纬度
  Longitude  — 经度
  Price      — 房价中位数（×10万美元）← 这是要预测的目标
"""
import pandas as pd
from sklearn.datasets import fetch_california_housing

# pd.set_option("display.max_columns",None)

raw = fetch_california_housing()
df = pd.DataFrame(raw.data, columns=raw.feature_names)
df["Price"] = raw.target  # 房价（单位：10万美元）

print("=== 前 5 行 ===")
print(df.head())

print(f"\n总共 {len(df)} 行，{len(df.columns)} 列")
print(f"\n=== 列名 ===")
for col in df.columns:
    print(f"  {col}")
