import numpy as np
import matplotlib.pyplot as plt

# 定义激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def swish(x):
    return x / (1 + np.exp(-x))

# 生成数据，x轴范围从-5到5
x = np.linspace(-5, 5, 400)

# 应用激活函数
sigmoid_values = sigmoid(x)
relu_values = relu(x)
leaky_relu_values = leaky_relu(x)
swish_values = swish(x)

# 选取部分点进行标记，以便不同形状区分
marker_indices = np.linspace(0, len(x)-1, 15, dtype=int)

# 颜色定义
colors = ['red', 'blue', 'green', 'purple']

# 绘制激活函数，并保留颜色与标记形状
plt.figure(figsize=(10, 6))
plt.plot(x, sigmoid_values, label='Sigmoid', color=colors[0], linestyle='-')
plt.scatter(x[marker_indices], sigmoid_values[marker_indices], marker='o', color=colors[0], label='Sigmoid Markers')  # 圆形

plt.plot(x, relu_values, label='ReLU', color=colors[1], linestyle='--')
plt.scatter(x[marker_indices], relu_values[marker_indices], marker='s', color=colors[1], label='ReLU Markers')  # 方形

plt.plot(x, leaky_relu_values, label='Leaky ReLU', color=colors[2], linestyle='-.')
plt.scatter(x[marker_indices], leaky_relu_values[marker_indices], marker='^', color=colors[2], label='Leaky ReLU Markers')  # 三角形

plt.plot(x, swish_values, label='Swish', color=colors[3], linestyle=':')
plt.scatter(x[marker_indices], swish_values[marker_indices], marker='d', color=colors[3], label='Swish Markers')  # 菱形

# 添加标签和标题
plt.title('Activation Functions with Different Colors and Markers')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)

# 设置x轴范围
plt.xlim(-5, 5)
plt.ylim(-0.5, 2)

# 添加图例
plt.legend()

# 显示图形
plt.show()
