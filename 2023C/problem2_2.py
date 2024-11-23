import numpy as np
import math
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import geatpy as ea

# 提取数据
cate_list = ['花叶类', '花菜类', '水生根茎类', '茄类', '辣椒类', '食用菌']
data21 = pd.DataFrame(pd.read_excel('result/problem2/品类单日加权定价统计.xlsx'))
data22 = pd.DataFrame(pd.read_excel('result/problem1/单日品类销售统计.xlsx'))
data23 = pd.DataFrame(pd.read_excel('result/problem2/品类加权损耗率统计.xlsx'))
data24 = pd.DataFrame(pd.read_excel('result/problem2/品类加权进价统计.xlsx'))


# 预测单类后七天的数据
def single_predict(data, start):
    pred = []
    sequence = data.iloc[start, 2:]
    X = range(1, 1086)
    X_2d = [[x] for x in X]
    sequence = sequence.fillna(sequence.mean())
    model = RandomForestRegressor(random_state=start, max_depth=5)
    model.fit(X_2d, sequence)
    pred.append(model.predict([[1086], [1087], [1088], [1089], [1090], [1091], [1092]]))
    return [item for sublist in pred for item in sublist]


# 均分列表
def split_into_sublists(input_list, sublist_size):
    big_list = []
    for i in range(0, len(input_list), sublist_size):
        sublist = input_list[i:i + sublist_size]
        big_list.append(sublist)
    return big_list


# 拟合系数
alpha = [[0.169674, -5.177816], [0.186577, -4.010956], [0.337228, -5.668421],
         [0.06551, -0.33593], [0.087314, -1.279035]]


# 定义销售量变化与定价变动函数
def Atan(x):
    if x < 1.376:
        return math.tan(x) * (1 / math.pi)
    else:
        return -0.3


# 预测各品类七天进价
C = []
for i in range(data24.shape[0]):
    C.append(single_predict(data24, i))
C = np.array(C).T.tolist()

# 预测花叶类七天定价和销售量
P_leaves = single_predict(data21, 0)
Q_leaves = single_predict(data22, 0)
print(P_leaves)
print(Q_leaves)

# 提取6月30日数据，作为初始值
P0 = list(data21.iloc[1:, data21.shape[1] - 5])
Q0 = list(data22.iloc[1:, 1086])

# 设置常数
loss = list(data23.iloc[:, 1])
for i in range(len(loss)):
    loss[i] = loss[i] / 100
discount = 0.6
n = 77


# 遗传算法
# 目标函数构建
@ea.Problem.single
def evalVars(Vars):
    P = Vars[:35]  # 定价
    M = Vars[35:]  # 进货量
    S = [[] for _ in range(7)]  # 库存量
    P = split_into_sublists(P, 5)
    M = split_into_sublists(M, 6)
    Q = [[] for _ in range(7)]  # 进价
    Pr = [[] for _ in range(7)]  # 利润
    for i in range(0, 7):
        if i == 0:
            Q[0].append(Q_leaves[0])
            for j in range(0, 5):
                q = Q0[j] * (1 + Atan((P[0][j] - P0[j]) / P0[j]))
                Q[0].append(q)
            for k in range(0, 6):
                S_k = M[0][k] - Q[0][k]
                if S_k > 0:
                    S[0].append(S_k)
                else:
                    S[0].append(0)
            for m in range(0, 6):
                rate = 1 - loss[m] + loss[m] * discount
                cost = C[0][m] * M[0][m]
                if m == 0:
                    if Q[0][m] < 0:
                        Pr[0].append(Q[0][m] * P_leaves[0] * discount - cost)
                    elif S[0][m] > 0 and Q[0][m] > 0:
                        Pr[0].append(Q[0][m] * P_leaves[0] * rate - cost)
                    elif S[0][m] <= 0:
                        Pr[0].append(M[0][m] * P_leaves[0] * rate - cost)
                    else:
                        Pr[0].append(0)
                else:
                    if Q[0][m] < 0:
                        Pr[0].append(Q[0][m] * P[0][m - 1] * discount - cost)
                    elif S[0][m] > 0 and Q[0][m] > 0:
                        Pr[0].append(Q[0][m] * P[0][m - 1] * rate - cost)
                    elif S[0][m] <= 0:
                        Pr[0].append(M[0][m] * P[0][m - 1] * rate - cost)
                    else:
                        Pr[0].append(0)
        else:
            Q[i].append(Q_leaves[i])
            for j in range(0, 5):
                q = Q[i - 1][j + 1] * (1 + Atan((P[i][j] - P[i - 1][j]) / P[i - 1][j]))
                if (q - Q[i - 1][j + 1]) / Q[i - 1][j + 1] > 2:
                    q = 1 * Q[i - 1][j + 1]
                Q[i].append(q)
            for k in range(0, 6):
                S_k = M[i][k] - Q[i][k]
                if S_k > 0:
                    S[i].append(S_k)
                else:
                    S[i].append(0)
            for m in range(0, 6):
                rate = 1 - loss[m] + loss[m] * discount
                cost = C[i][m] * M[i][m]
                flag = S[i - 1][m] + M[i][m] - Q[i][m]
                if m == 0:
                    if Q[i][m] < 0:
                        Pr[i].append(0)
                    elif S[i - 1][m] > Q[i][m]:
                        Pr[i].append(Q[i][m] * P_leaves[i] * discount - cost)
                    elif flag > 0 and S[i - 1][m] < Q[i][m]:
                        Pr[i].append(
                            S[i - 1][m] * P_leaves[i] * discount + (Q[i][m] - S[i - 1][m]) * P_leaves[i] * rate - cost)
                    elif flag <= 0:
                        Pr[i].append(S[i - 1][m] * P_leaves[i] * discount + M[i][m] * P_leaves[i] * rate - cost)
                    else:
                        Pr[i].append(0)
                else:
                    if Q[i][m] < 0:
                        Pr[i].append(0)
                    elif S[i - 1][m] > Q[i][m]:
                        Pr[i].append(Q[i][m] * P[i][m - 1] * discount - cost)
                    elif flag > 0 and S[i - 1][m] < Q[i][m]:
                        Pr[i].append(
                            S[i - 1][m] * P[i][m - 1] * discount + (Q[i][m] - S[i - 1][m]) * P[i][m - 1] * rate - cost)
                    elif flag <= 0:
                        Pr[i].append(S[i - 1][m] * P[i][m - 1] * discount + M[i][m] * P[i][m - 1] * rate - cost)
                    else:
                        Pr[i].append(0)
    f = np.sum(np.array(Pr))
    return f


# 约束条件构建
D = []
for sublist in C:
    for item in sublist:
        D.append(item * 0.5)

D = D[7:]

types = [0] * n
lb = D + [0] * 42
ub = [20] * 35 + [120] * 42

# 问题参数设置
problem = ea.Problem(name='solver',
                     M=1,
                     maxormins=[-1],
                     Dim=n,
                     varTypes=types,
                     lb=lb,
                     ub=ub,
                     evalVars=evalVars)

# 调用增强精英保留策略的遗传算法
algorithm = ea.soea_SEGA_templet(problem,
                                 ea.Population(Encoding='BG', NIND=300),
                                 MAXGEN=200,
                                 logTras=1,
                                 trappedValue=1e-5,
                                 maxTrappedCount=10)

# 求解
res = ea.optimize(algorithm, seed=1, verbose=True, drawing=1, outputMsg=True, drawLog=False, saveFlag=True,
                  dirName='result/problem2/solver')
