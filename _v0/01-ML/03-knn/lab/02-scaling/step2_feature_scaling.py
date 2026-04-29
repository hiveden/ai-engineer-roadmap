# 特征缩放 feature scaling - KNN 必备前置步骤
# 为什么需要：KNN/SVM/神经网络都靠"距离 or 梯度"决策，量纲大的特征会霸占贡献
# 两种主流方案：归一化 normalize（MinMax）/ 标准化 standardize（Z-score）

from sklearn.preprocessing import MinMaxScaler, StandardScaler


def dm01_MinMaxScaler():
    # 归一化：把每列压到 [0, 1]，公式 (x - col_min) / (col_max - col_min)
    # 默认范围 [0,1]，可改 feature_range=(-1, 1)
    data = [[90, 2, 10, 40],
            [60, 4, 15, 45],
            [75, 3, 13, 46]]

    transformer = MinMaxScaler()

    # fit_transform = fit（按列算 min/max）+ transform（套公式）
    # 注意：测试集只能 transform，不能 fit_transform（避免 data leakage）
    data = transformer.fit_transform(data)
    print(data)


def dm03_StandardScaler():
    # 标准化（Z-score）：把每列变成 mean=0, std=1，公式 (x - col_mean) / col_std
    # 不限范围，对异常值比 MinMax 鲁棒
    data = [[90, 2, 10, 40],
            [60, 4, 15, 45],
            [75, 3, 13, 46]]

    transformer = StandardScaler()
    data = transformer.fit_transform(data)
    print(data)

    # fit 后学到的统计量，每列一个值
    print('transformer.mean_ -->', transformer.mean_)  # 每列均值
    print('transformer.var_  -->', transformer.var_)   # 每列方差（注意：是 var 不是 std！）


if __name__ == '__main__':
    dm01_MinMaxScaler()
    dm03_StandardScaler()
