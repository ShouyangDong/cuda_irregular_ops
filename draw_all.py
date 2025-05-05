import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 读取CSV数据，处理多层次列标题
df = pd.read_csv('2.csv', header=[0, 1])

# 2. 打印原始多级列名以调试
print("原始多级列名:")
print(df.columns)

# 3. 平坦化多级列名，将其合并为单级列名，忽略 'Unnamed' 部分
new_columns = []
last_main = ''

for col in df.columns:
    if 'Unnamed' not in col[0]:
        last_main = col[0]
        if 'Unnamed' in str(col[1]) or pd.isna(col[1]):
            new_columns.append(col[0])
        else:
            new_columns.append(f"{col[0]} {col[1]}")
    else:
        if 'Unnamed' in str(col[1]) or pd.isna(col[1]):
            new_columns.append(col[0])
        else:
            new_columns.append(f"{last_main} {col[1]}")

df.columns = new_columns

# 4. 打印平坦化后的列名以确认
print("\n平坦化后的列名:")
print(df.columns)

# 5. 设置全局字体大小和图表参数
plt.rcParams.update({
    'font.size': 16,             # 全局字体大小
    'axes.labelsize': 18,        # 轴标签字体大小
    'xtick.labelsize': 14,       # X轴刻度字体大小
    'ytick.labelsize': 14,       # Y轴刻度字体大小
    'legend.fontsize': 20,       # 图例字体大小 (增大)
    'figure.figsize': (20, 15)   # 图表尺寸
})

# 6. 创建四个子图，纵向排列，共享X轴，使用 constrained_layout=False
fig, axes = plt.subplots(4, 1, figsize=(20, 15), sharex=True, constrained_layout=False)

fig.patch.set_facecolor('white')  # 设置背景为白色

# 7. 创建布尔掩码，包含所有行，包括“Overall”
mask = df['Type'].notna()

# 8. 获取算子类型，包括“Overall”
operators = df.loc[mask, 'Type']
x = np.arange(len(operators))  # 算子的数量

# 9. 在最后一组数据前增加空隙
x_new = x.copy()
if len(x_new) > 1:
    x_new[-1] += 1  # 将最后一组数据向右移动1单位，创建空隙

# 10. 柱状图的宽度
width = 0.35

# 11. PyTorch性能固定为1
pytorch_perf = [1] * len(operators)

# 12. 定义转换类型及其对应的列名和转换名称
transitions = ['C->CUDA', 'CUDA->BANG', 'CUDA->HIP', 'CUDA->C']
transition_names = [
    'C with VNNI → CUDA C',
    'CUDA C → BANG C',
    'CUDA C → HIP',
    'CUDA C → C with VNNI'
]

# 13. 初始化图例句柄和标签
handles = []
labels = []

# 14. 迭代绘制每个子图
for idx, (transition, transition_name) in enumerate(zip(transitions, transition_names)):
    ax = axes[idx]

    # 15. 构造列名
    corrected_cases_col = f'{transition} Corrected cases'
    speedup_col = f'{transition} SpeedUp Over Pytorch'

    # 16. 检查列是否存在，避免KeyError
    if corrected_cases_col not in df.columns or speedup_col not in df.columns:
        print(f'列 "{corrected_cases_col}" 或 "{speedup_col}" 未找到。请检查列名。')
        continue

    # 17. 获取对应转换的 Corrected cases 和 SpeedUp Over Pytorch，包含“Overall”行
    corrected_cases = df.loc[mask, corrected_cases_col].fillna(0).values
    speedup = df.loc[mask, speedup_col].fillna(0).values

    # 18. 检查数据长度是否匹配
    if len(speedup) != len(x_new):
        print(f'警告: "{transition} SpeedUp Over Pytorch" 的长度 {len(speedup)} 与 operators 的长度 {len(x_new)} 不匹配。')
        continue

    # 19. 绘制柱状图
    bars1 = ax.bar(x_new - width / 2, pytorch_perf, width, label='PyTorch', color='orange', zorder=2)  # PyTorch为橙色
    bars2 = ax.bar(x_new + width / 2, speedup, width, label='QiMeng-Xpiler', color='purple', zorder=2)        # Falcon为紫色

    # 20. 设置X轴刻度
    ax.set_xticks(x_new)
    ax.set_xticklabels(operators, rotation=45, ha='right')

    # 21. 添加网格线
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5)

    # 22. 创建第二个Y轴用于绘制折线图
    ax2 = ax.twinx()
    line = ax2.plot(x_new, corrected_cases, color='red', marker='o', label='Corrected Cases', zorder=3)[0]  # Corrected Cases为红色

    # 23. 设置右y轴固定范围并设置刻度
    ax2.set_ylim(0, 10)
    ax2.set_yticks([0, 8])

    # 24. 移除折线点数据标签
    # (无注释)

    # 25. 添加子图标签和转换名称
    ax.text(0.5, 1.05, f"({chr(97+idx)}) {transition_name}", transform=ax.transAxes, fontsize=20, fontweight='bold', ha='center')

    # 26. 仅在第一个子图中收集图例句柄
    if idx == 0:
        handles.extend([bars1[0], bars2[0], line])
        labels.extend(['PyTorch', 'QiMeng-Xpiler', 'Corrected Cases'])

# 27. 创建统一的图例，位于整个图表的顶部中心，增大字体
fig.legend(handles=handles, labels=labels, loc='upper center', fontsize=24, ncol=3)

# 28. 添加共享的左侧和右侧纵坐标标题，并确保其位于图表外侧
fig.text(0.02, 0.5, 'Normalized Performance', va='center', rotation='vertical', fontsize=18, ha='center')
fig.text(0.98, 0.5, 'Corrected Cases', va='center', rotation='vertical', fontsize=18, ha='center')

# 29. 调整整体布局，确保图例和子图标签不被遮挡
# 保持用户指定的边距
fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.2, hspace=0.3)

# 30. 显示图表
#plt.show()
fig.savefig("Figure_7-3_new.pdf")
