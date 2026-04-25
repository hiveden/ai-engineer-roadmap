# GridSearchCV 网格搜索 - 朴素版（含 data leakage 坑）
# ⚠️ 这份代码是反面教材！
# 看起来对，跑出来 best_score_ ≈ 0.971 比正确版的 0.933 还高
# 但那 4% 是 data leakage 偷来的——上线后真实表现会比 0.933 还差
#
# 坑在哪：
#   train_test_split 之后立刻 fit_transform 了整个训练集
#   GridSearchCV 内部又把训练集切 train/val 做交叉验证
#   但 scaler 已经在"完整训练集"上 fit 过了
#   → 验证集（每折 25%）的统计量泄漏给了 scaler → 评估虚高
#
# 正确版见 step6_cv_gridsearch_pipeline.py

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# 1. 加载数据
mydataset = load_iris()

# 2. 切分（注意没用 stratify，凑巧 iris 类别均衡所以问题不大）
x_train, x_test, y_train, y_test = train_test_split(
    mydataset.data, mydataset.target, test_size=0.3, random_state=22, stratify=mydataset.target
)

# 3. 特征工程
# ⚠️ 这里有坑：在 GridSearchCV 之前 fit_transform 了
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)   # ← 这步看到了"完整训练集"的统计量
x_test = transfer.transform(x_test)

# 4. 网格搜索 + 交叉验证
model = KNeighborsClassifier()
param_dict = {'n_neighbors': [1, 3, 5, 7, 9, 11]}

# cv=4 表示 4 折交叉验证
# GridSearchCV 内部会把 x_train 再切成 4 份做 train/val
# 但 x_train 已经是标准化过的，val 折的统计量已经被 scaler 看过了 → leakage
estimator = GridSearchCV(model, param_dict, cv=4)
estimator.fit(x_train, y_train)

print('best_score_:    ', estimator.best_score_)      # 大约 0.971（虚高，含 leakage）
print('best_params_:   ', estimator.best_params_)     # {'n_neighbors': 7}
print('best_estimator_:', estimator.best_estimator_)
# print(estimator.cv_results_)   # 详细每折结果

# 5. 用最佳模型在测试集评估
estimator_ = estimator.best_estimator_
y_predict = estimator_.predict(x_test)
print('test accuracy:  ', accuracy_score(y_test, y_predict))
