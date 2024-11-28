from z3 import *

# 创建 Z3 求解器
solver = Solver()

# 原始循环和拆分后的循环变量
result_original = Int("result_original")
result_split = Int("result_split")

# 定义数组 A，用于保存迭代过程中的值 (模拟累加)
A = Function("A", IntSort(), IntSort())

# 原始循环: result += i, i 从 0 到 9
result = Int("result")
solver.add(result == 0)  # 初始值为 0
sum_original = result
for i in range(10):
    sum_original += i
solver.add(result_original == sum_original)

# 拆分后的嵌套循环: 外层循环从 0 到 1，内层循环从 0 到 4
sum_split = result
for i in range(2):
    for j in range(5):
        sum_split += (i * 5) + j
solver.add(result_split == sum_split)

# 验证原始循环和拆分后的嵌套循环是否等效
solver.add(result_original != result_split)

# 检查是否有解，如果没有解则表示等效
if solver.check() == sat:
    print("循环拆分后的版本与原始版本不等效。")
else:
    print("循环拆分后的版本与原始版本等效。")
