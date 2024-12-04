from z3 import *

# 创建 Z3 求解器
solver = Solver()

# 定义变量 n 代表内层 k 的上界
n = Int("n")

# 原始的总迭代次数
original_iteration_count = 256 * 1024

# 拆分后的总迭代次数
# 外层循环 256 次，j 从 0 到 4，k 从 0 到 n
split_iteration_count = 256 * 4 * n

# 添加约束：原始总迭代次数等于拆分后的总迭代次数
solver.add(original_iteration_count == split_iteration_count)

# 检查是否有解并输出
if solver.check() == sat:
    model = solver.model()
    print(f"内层循环 k 的上界为: {model[n]}")
else:
    print("无法找到等效的拆分方案。")
