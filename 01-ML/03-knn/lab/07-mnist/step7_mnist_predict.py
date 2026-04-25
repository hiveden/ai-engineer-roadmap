# MNIST 手写数字识别 - 推理
# 流程：读图 → 预处理 → 加载模型 → 预测
#
# 工程坑预警：plt.imread 对 PNG / JPG 行为不一致！
# - PNG（带 alpha 通道）：自动归一化到 [0, 1]，shape 是 (H, W, 4)
# - JPG（无 alpha）：保留 [0, 255]，shape 是 (H, W, 3)
# - 灰度图：可能是 (H, W) 单通道
# 真实场景的图片必须**手动统一格式**，不能闭眼信 imread

import matplotlib.pyplot as plt
import joblib


# 假设有一张 28×28 灰度 PNG（已经手动准备好的 demo 图）
path = '../../../../assets/source-materials/demo.png'

# 读图：返回 ndarray
x = plt.imread(path)
# x.shape 可能是：
#   (28, 28)         单通道灰度
#   (28, 28, 3)      RGB 彩图
#   (28, 28, 4)      RGBA 彩图
# print(x, x.shape)

# 显示看看
plt.imshow(x, cmap='gray')
# plt.show()

# 预处理：把 28×28 reshape 成 (1, 784) 才能喂给 model.predict
x = x.reshape(1, -1)   # -1 = 自动推断（这里推断为 784）
print(x.shape)         # (1, 784)

# 加载训练好的模型
model = joblib.load('手写数字识别.pth')

# 推理
y_predict = model.predict(x)
print(y_predict)


# ============================================================
# 真实场景的图（如手机拍的、截图来的）需要更多预处理：
# ============================================================
'''
import numpy as np
from PIL import Image

# 1. 加载（用 PIL 比 plt.imread 行为更稳定）
img = Image.open(path).convert('L')   # 'L' = 灰度

# 2. resize 到 28×28
img = img.resize((28, 28))

# 3. 转 ndarray
x = np.array(img)
print(x.shape, x.dtype)   # (28, 28) uint8 [0, 255]

# 4. 归一化（如果训练时用了 x / 255，这里也要）
x = x / 255.0

# 5. reshape 成 (1, 784)
x = x.reshape(1, -1)

# 6. 推理
y = model.predict(x)
'''
