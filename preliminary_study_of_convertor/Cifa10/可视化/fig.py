import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')

# 数据
data_group1 = [0.97, 0.95, 0.92]  # 第一组数据
data_group2 = [0.10, 0.89, 0.88]  # 第二组数据

# 设置组的标签
labels = ['Resnet18', 'Resnet50', 'Resnet101']

# 设置每个柱状图的宽度
bar_width = 0.2

# 设置柱状图的位置
bar_positions_group1 = np.arange(len(labels))
bar_positions_group2 = bar_positions_group1 + bar_width


# 绘制第一组柱状图
plt.bar(bar_positions_group1, data_group1, width=bar_width, label='ANN',color = 'Turquoise')
# 绘制第二组柱状图
plt.bar(bar_positions_group2, data_group2, width=bar_width, label='Converted SNN',color = 'PaleVioletRed')


# 添加标签和标题
plt.xlabel('Name of Neural Network')
plt.ylabel('Accuracy on Cifa10')
plt.title('Comparison of Different Neural Networks on CIFA10')
plt.xticks(bar_positions_group1 + bar_width / 2, labels)  # 设置X轴刻度位置为两组柱状图的中间位置
plt.legend()  # 显示图例

# 显示图形
plt.show()

