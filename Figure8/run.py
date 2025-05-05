import numpy as np
import matplotlib.pyplot as plt

# 运算符名称，较长的名称通过添加换行符 '\n' 分行显示
operators = ['Relu', 'Softmax', 'GEMM', 'Conv2D\nNHWC', 'Self\nAttention', 'Deformable\nAttention']

# 随机生成的数据，每个运算符4个时间值
data = np.random.rand(6, 5) * 100
data[0, 0] = 2834.218
data[0, 1] = 636.364
data[0, 2] = 0
data[0, 3] = 364.470
data[0, 4] = 280.707

data[1, 0] = 2834.218 * 8 /3
data[1, 1] = 636.364
data[1, 2] = 200.143
data[1, 3] = 364.470 * 1.1
data[1, 4] = 280.707 * 1.03


data[2, 0] = 2834.218 * 4 / 3
data[2, 1] = 636.364
data[2, 2] = 200.143 * 2 * 4
data[2, 3] = 364.470 * 10
data[2, 4] = 280.707 * 1.1

data[3, 0] = 2834.218 * 4 /3
data[3, 1] = 636.364
data[3, 2] = 200.143 * 2 * 6
data[3, 3] = 364.470 * 14
data[3, 4] = 280.707 * 1.03

data[4, 0] = 2834.218 * 15 / 3
data[4, 1] = 636.364 * 2.0
data[4, 2] = 200.143 * 2 * 6 * 2
data[4, 3] = 364.470 * 10 * 2
data[4, 4] = 280.707 * 2.0 

data[5, 0] = 2834.218 * 1
data[5, 1] = 636.364
data[5, 2] = 200.143 * 2 * 30
data[5, 3] = 364.470
data[5, 4] = 280.707 * 1.05


data = np.round(data / 3600, 1)
# 创建一个图和一个子图，调整图的尺寸
fig, ax = plt.subplots(figsize=(10, 6))  # 增大图表尺寸以适应换行后的标签

# 条形图的宽度
bar_width = 0.6

# 为不同的项目设置不同的x位置
x = np.arange(len(operators))

# 绘制堆叠条形图
bars1 = ax.bar(x, data[:, 0], width=bar_width, label='LLM')
bars2 = ax.bar(x, data[:, 1], width=bar_width, bottom=data[:, 0], label='Unit Test')
bars3 = ax.bar(x, data[:, 2], width=bar_width, bottom=data[:, 0] + data[:, 1], label='SMT')
bars4 = ax.bar(x, data[:, 3], width=bar_width, bottom=data[:, 0] + data[:, 1] + data[:, 2], label='Autotuning')
bars5 = ax.bar(x, data[:, 4], width=bar_width, bottom=data[:, 0] + data[:, 1] + data[:, 2]+ data[:, 3], label='Evaluation')
# 计算总时间并添加文本标签
total_times = data.sum(axis=1)
for idx, rect in enumerate(bars5):
    height = rect.get_height() + rect.get_y()
    ax.text(rect.get_x() + rect.get_width() / 2, height, f'{total_times[idx]:.1f}', ha='center', va='bottom', fontsize=12)

# 设置x轴标签和图例
# ax.set_xlabel('Operators')
ax.set_ylabel('Time(Hours)', fontsize=14)
ax.set_title("Breakdown of QiMeng-Xpiler's Compilation Time", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(operators, ha='center', fontsize=14)  # 设置水平对齐方式为居中

# 计算总时间的平均值并绘制灰色虚线
average_time = total_times.mean()
ax.axhline(y=average_time, color='gray', linestyle='--', linewidth=1, label='Average Time')

# 在灰色虚线的最左侧添加对应的值
ax.text(1, average_time, f'Average: {average_time:.1f}', color='gray', va='bottom', ha='left', fontsize=12)

# 调整图例
ax.legend(fontsize=12, loc='upper left', bbox_to_anchor=(1, 1), ncol=1)

# 旋转x轴标签，以便它们不会重叠
plt.xticks(rotation=45)

# 显示图表
plt.tight_layout(pad=3)  # 调整布局以防止剪切和重叠
plt.show()
