import matplotlib.pyplot as plt
import numpy as np

# 数据
operators = ["falcon w/o Auto-tuning", "falcon", "TVM"]
targets = ["CUDA C", "BANG C", "Hip", "DL Boost"]

matmul_performance = {
    "falcon w/o Auto-tuning": [1.00, 1.00, 1.00, 1.00],
    "falcon": [2.01, 4.70, 3.87, 4.82],
    "TVM": [3.19, 4.85, 5.14, 4.93],
}

conv2d_performance = {
    "falcon w/o Auto-tuning": [1.00, 1.00, 1.00, 1.00],
    "falcon": [9.23, 3.04, 3.97, 3.01],
    "TVM": [13.50, 3.00, 5.41, 3.12],
}

# 创建 Matmul 和 Conv2d 性能对比并排柱状图
fig, ax = plt.subplots(2, 1, figsize=(6, 6))

bar_width = 0.25
index = np.arange(len(targets))

for i, operator in enumerate(operators):
    rects = ax[0].bar(
        index + i * bar_width,
        matmul_performance[operator],
        bar_width,
        label=operator,
    )
    for rect in rects:
        height = rect.get_height()
        ax[0].annotate(
            "{}".format(height),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

ax[0].set_xlabel("Targets")
ax[0].set_ylabel("Performance (microseconds)")
ax[0].set_title("(a) Matmul Performance Comparison")
ax[0].set_xticks(index + bar_width / 2)
ax[0].set_xticklabels(targets)
ax[0].legend()
ax[0].set_yticks(np.arange(0, 20, 5))
ax[0].set_ylim(0, 20)  # Adjust the y-axis limits for better visibility

for i, operator in enumerate(operators):
    rects = ax[1].bar(
        index + i * bar_width,
        conv2d_performance[operator],
        bar_width,
        label=operator,
    )
    for rect in rects:
        height = rect.get_height()
        ax[1].annotate(
            "{}".format(height),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

ax[1].set_xlabel("Targets")
ax[1].set_ylabel("Performance (microseconds)")
ax[1].set_title("(b) Conv2d Performance Comparison")
ax[1].set_xticks(index + bar_width / 2)
ax[1].set_xticklabels(targets)
ax[1].legend()
ax[1].set_yticks(np.arange(0, 15, 3))
ax[1].set_ylim(0, 15)  # Adjust the y-axis limits for better visibility

plt.tight_layout()
plt.show()
