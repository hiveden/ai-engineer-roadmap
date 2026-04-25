# 鸢尾花 KNN 端到端最小例
# 标准 6 步流程：加载 → 切分 → 特征工程 → 训练 → 评估 → 推理
# 这是 sklearn 监督学习的"模板"，所有分类/回归任务都能套这个框架

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def dm01_iris_end_to_end():
    # ---- 1. 加载数据 ----
    mydataset = load_iris()

    # ---- 2. 数据预处理：切分 ----
    # test_size=0.3 / random_state=22 / stratify 分层抽样保比例
    x_train, x_test, y_train, y_test = train_test_split(
        mydataset.data, mydataset.target,
        test_size=0.3, random_state=22, stratify=mydataset.target,
    )

    # ---- 3. 特征工程：标准化 ----
    transfer = StandardScaler()
    # 训练集 fit_transform：算 mean/std 并变换
    x_train = transfer.fit_transform(x_train)
    # 测试集只 transform：复用训练集学到的 mean/std（避免 data leakage）
    x_test = transfer.transform(x_test)

    # ---- 4. 模型训练 ----
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(x_train, y_train)

    # ---- 5. 模型评估 ----
    y_predict = model.predict(x_test)
    print('预测结果：', y_predict)
    print('真实标签：', y_test)

    # 两种打分方式（结果完全一样，选其一）
    myscore1 = accuracy_score(y_test, y_predict)         # 函数式：传 y_true, y_pred
    myscore2 = model.score(x_test, y_test)               # 方法式：传 X_test, y_test，内部自己 predict
    print('准确率1（accuracy_score）：', myscore1)
    print('准确率2（model.score）：', myscore2)

    # ---- 6. 模型推理：来一条新数据 ----
    x_new = [[3, 5, 4, 2]]                                # 必须 2D（哪怕只一条）
    x_new = transfer.transform(x_new)                     # 推理也要标准化（用同一个 transfer）
    y_new = model.predict(x_new)
    print('新数据预测：', y_new)

    # predict_proba：分类专属，返回每类概率
    # KNN 的"概率" = k 个邻居中各类占比（不是真概率，但能用来做阈值调节）
    y_new_proba = model.predict_proba(x_new)
    print('概率分布：', y_new_proba)


if __name__ == '__main__':
    dm01_iris_end_to_end()
