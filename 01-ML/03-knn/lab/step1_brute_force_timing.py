from sklearn.datasets import load_digits

X, y = load_digits(return_X_y = True)
print(f"X.shape = {X.shape}")
print(f"y[:20] = {y[:20]}")

from sklearn.model_selection import train_test_split

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"X_tr.shape = {X_tr.shape}")

from sklearn.neighbors import KNeighborsClassifier
import time

model = KNeighborsClassifier(n_neighbors=5)

t0 = time.time()
model.fit(X_tr, y_tr)
t1 = time.time()
print(f"训练耗时: {(t1-t0)* 1000:.2f} ms")

t0 = time.time()
y_pred = model.predict(X_te)
t1 = time.time()

predict_ms = (t1-t0) * 1000
print(f"预测耗时：{predict_ms:.2f} ms (360 条测试样本)")
