# 数据集分割 + 分层抽样 stratify
# train_test_split：sklearn 最常用的工具之一，分类任务必加 stratify

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def dm01_split_basic():
    mydataset = load_iris()

    # train_test_split 关键参数：
    # - test_size: 测试集比例（也可写整数表示样本数）
    # - random_state: 随机种子，固定后切分结果可复现
    # - stratify: 分层抽样按这一列的类别比例切，分类任务默认应该开
    x_train, x_test, y_train, y_test = train_test_split(
        mydataset.data,
        mydataset.target,
        test_size=0.2,
        random_state=10,
        stratify=mydataset.target,
    )

    # iris 类别均衡（50/50/50），stratify 保证：
    # 训练集每类 40 个 / 测试集每类 10 个（150 × 0.2 = 30 测试集，平均到 3 类）
    print(x_train.shape)  # (120, 4)
    print(x_test.shape)   # (30, 4)
    print(y_train.shape)  # (120,)
    print(y_test.shape)   # (30,)


if __name__ == '__main__':
    dm01_split_basic()
