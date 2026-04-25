# KNN 算法 API 使用 - 分类问题
# sklearn 三件套：构造 → fit → predict
# KNN 是 lazy learner：fit 只存数据，predict 时才算距离/投票。

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


def dm01_knn_api_分类():
    # 超参 k：看最近几个邻居投票。k=1 最容易过拟合，生产常用 3/5/7（奇数避免平票）
    model = KNeighborsClassifier(n_neighbors=1)

    # sklearn 硬规矩：X 必须 2D [n_samples, n_features]，y 1D [n_samples]
    # 类比：X 是数据库表（行=样本，列=字段），y 是单独一列 label
    X = [[0], [1], [2], [3]]   # 4 样本 × 1 特征
    y = [0, 0, 1, 1]

    model.fit(X, y)        # KNN 的 fit ≈ 把训练集存进对象，几乎 O(1)

    # predict 是批量接口，输入也必须 2D，哪怕只查一个也要包成 batch
    myret = model.predict([[4]])
    print('myret-->', myret)

    # x=4 → 最近邻是 3（距离 1）→ 标签 1 → 预测 [1]


def dm02_knn_api_回归():
    # 分类 → 回归 只换两件事：Classifier 改 Regressor，y 从离散标签变连续值
    # 预测逻辑也变了：分类是"邻居投票"，回归是"邻居 y 取平均"
    model = KNeighborsRegressor(n_neighbors=2)

    X = [[0, 0, 1],            # 4 样本 × 3 特征
         [1, 1, 0],
         [3, 10, 10],
         [4, 11, 12]]
    y = [0.1, 0.2, 0.3, 0.4]   # 连续值（不是类别编号）

    model.fit(X, y)

    # 查询 [3,11,10]：欧氏距离最近的 2 个是 [3,10,10]→0.3 和 [4,11,12]→0.4
    # 预测 = (0.3 + 0.4) / 2 = 0.35
    myret = model.predict([[3, 11, 10]])
    print('myret-->', myret)


if __name__ == '__main__':
    dm01_knn_api_分类()
    dm02_knn_api_回归()
