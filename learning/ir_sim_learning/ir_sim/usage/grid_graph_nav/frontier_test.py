import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 生成示例数据
x = np.linspace(0, 10, 100)
y = np.linspace(0, 10, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) + np.cos(Y)

# 自定义颜色映射
colors = ['blue', 'green', 'red']
cmap = ListedColormap(colors)

# 绘制图像
plt.contourf(X, Y, Z, levels=[0, 1, 10], cmap=cmap)
plt.colorbar(label='Values')
plt.title('Custom Colors for Values 0, 1, and 10')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# 显示图形
plt.show()

