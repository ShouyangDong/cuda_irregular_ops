from z3 import *

# 定义数组的大小
N = 3  # 行数
M = 3  # 列数

# 定义符号变量矩阵 A_before 和 A_after
A_before = [[Int(f"A_before_{i}_{j}") for j in range(M)] for i in range(N)]
A_after = [[Int(f"A_after_{i}_{j}") for j in range(M)] for i in range(N)]

# 初始化 Solver
s = Solver()

# 定义初始条件，A_before 和 A_after 的初始状态一致
initial_conditions = [
    A_before[i][j] == A_after[i][j] for i in range(N) for j in range(M)
]
s.add(initial_conditions)

# 添加变换前的行为模型：for i in range(N): for j in range(M): A[i][j] += 1
for i in range(N):
    for j in range(M):
        s.add(A_before[i][j] == A_before[i][j] + 1)

# 添加变换后的行为模型：for j in range(M): for i in range(N): A[i][j] += 1
for j in range(M):
    for i in range(N):
        s.add(A_after[i][j] == A_after[i][j] + 1)

# 验证 A_before 和 A_after 是否相等
equivalence_conditions = [
    A_before[i][j] == A_after[i][j] for i in range(N) for j in range(M)
]
s.add(Not(And(*equivalence_conditions)))

# 检查等效性
if s.check() == sat:
    print("代码不等效，找到反例：", s.model())
else:
    print("代码等效")
