# GridSearchCV 网格搜索 - 正确版（用 Pipeline 防 leakage）
# ✅ 标准做法：把 scaler 和 model 串成 Pipeline，作为整体喂给 GridSearchCV
# 这样每折 CV 内部独立 fit scaler，验证集对 scaler 完全不可见 → 无 leakage
#
# 跟朴素版对比：
#   朴素版 best_score_ ≈ 0.971（虚高）
#   本版   best_score_ ≈ 0.933（真实）
#   差的 4% 就是 data leakage 在朴素版偷的"分数"

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline


# 1. 加载数据
mydataset = load_iris()

# 2. 切分
x_train, x_test, y_train, y_test = train_test_split(
    mydataset.data, mydataset.target, test_size=0.3, random_state=20
)

# 3. 构建 Pipeline：scaler + model 串起来
# Pipeline = 多个 step 的有序组合，每步都是 (name, transformer/estimator) 元组
# 调用 pipe.fit(X, y)：
#   step1: scaler.fit_transform(X)  → X_scaled
#   step2: model.fit(X_scaled, y)
# 调用 pipe.predict(X_new)：
#   step1: scaler.transform(X_new)  ← 注意只 transform
#   step2: model.predict(X_scaled)
pipe = Pipeline([
    ('std', StandardScaler()),                    # name 自定义
    ('knn2', KNeighborsClassifier()),
])

# 4. 网格搜索
# 参数命名规则：'<step_name>__<param_name>'
# 'knn2__n_neighbors' = pipe 里 'knn2' 这步的 n_neighbors 参数
# 双下划线是 sklearn 约定（必须双下划线，单下划线不识别）
param_dict = {'knn2__n_neighbors': [1, 3, 5, 7, 9, 11]}

# 关键：把 pipe 当做 estimator 喂进去
# GridSearchCV 内部每折切 train/val 时：
#   train 折：pipe.fit() → scaler 在 train 折上 fit + transform，model 在 scaled train 上训练
#   val 折：pipe.predict() → scaler 用 train 折学的统计量 transform val，model 预测
# val 折的统计量从未泄漏给 scaler → 无 leakage ✅
estimator = GridSearchCV(pipe, param_dict, cv=4)
estimator.fit(x_train, y_train)

print('best_score_:    ', estimator.best_score_)      # 大约 0.933（真实）
print('best_params_:   ', estimator.best_params_)
print('best_estimator_:', estimator.best_estimator_)

# 5. 用最佳模型在测试集评估
y_predict = estimator.predict(x_test)   # GridSearchCV 已经 refit 了，可以直接 predict
print('test accuracy:  ', accuracy_score(y_test, y_predict))
