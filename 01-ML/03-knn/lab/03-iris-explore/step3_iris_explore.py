# 鸢尾花数据集探索 + 可视化
# sklearn 自带的"机器学习 hello world" 数据集
# 150 样本 × 4 特征 × 3 类（setosa / versicolor / virginica）

from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def dm01_iris_view():
    # load_iris 返回 Bunch 对象（dict 子类，可属性访问也可 [key] 访问）
    mydataset = load_iris()

    # 关键 keys：data / target / target_names / feature_names / DESCR
    print(mydataset.keys())

    # 特征矩阵：shape (150, 4)，dtype float64
    print(mydataset.data, len(mydataset.data))

    # 标签：shape (150,)，整数 0/1/2 对应 target_names
    print(mydataset.target, mydataset.target_names)

    # 特征名：花萼长/宽 + 花瓣长/宽
    print(mydataset.feature_names)

    # DESCR 是数据集说明文档（探索阶段必看）
    print(mydataset.DESCR)


def dm02_iris_visualize():
    # numpy ndarray 不能直接喂 seaborn，要先转 DataFrame
    mydataset = load_iris()
    print(type(mydataset.data))   # <class 'numpy.ndarray'>

    # numpy → pandas：参 1 数据 / 参 2 列名
    data = pd.DataFrame(mydataset.data, columns=mydataset.feature_names)

    # 加一列 label（直接列赋值，pandas 自动按行对齐）
    data['label'] = mydataset.target
    print(data)

    # seaborn lmplot：散点图 + 可选拟合线
    # hue='label' 按类别上色（这是分类可视化的标配套路）
    # fit_reg=False 关掉默认的回归拟合线
    sns.lmplot(data=data,
               x='petal length (cm)',
               y='petal width (cm)',
               hue='label',
               fit_reg=False)

    plt.title('Iris data')
    plt.show()


if __name__ == '__main__':
    dm01_iris_view()
    dm02_iris_visualize()
