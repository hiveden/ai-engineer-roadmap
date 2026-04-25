"""
案例: 演示 KNN算法 识别图片, 即: 手写数字识别案例.

介绍:
    每张图片都是由 28 * 28 像素组成的, 即: 我们的csv文件中每一行都有 784个像素点, 表示图片(每个像素)的 颜色.
    最终构成图像.

1  图像加载
2  图像预处理
3  模型载入
4  模型预测
"""
import matplotlib.pyplot as plt
import joblib

#1  图像加载
path=r'D:\AI_itheima\class\20260423_Shanghai_Fundamentals_of_Machine_Learning\code\datasets\demo.png'
# 利用 plt.imread读取 图片， 如果是 png格式，则自动归一化  如果是 jpg，则不会进行归一化

# image  read  输入路径，得到读取后的图像
x = plt.imread(path)
#print(x, x.shape)

# 显示图片
plt.imshow(x, cmap='gray')
#plt.show()

# 2  图像预处理，转换数据的形状，从28x28，转化为784列
x = x.reshape(1, -1)  #  -1 代表自动推断列数
print(x.shape)   # (1, 784)

# 载入训练好的模型
model = joblib.load('手写数字识别.pth')

# 推理训练好的模型
y_predict = model.predict(x)
print(y_predict)

#0,1,2,3,4,5,6,7,8~9
'''
x = plt.imread(path) #  2048*1080
x = x.reshape(28,28)  # reshape :28*28
# 将3通道的彩图 转为1通道 
x = x[:,:,0]     #  H , W ,  C   
#判断是否需要归一化

# 2  图像预处理，转换数据的形状，从28x28，转化为784列
x = x.reshape(1, -1)  #  -1 代表自动推断列数
'''