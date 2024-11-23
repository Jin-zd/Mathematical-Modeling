import math
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import geatpy as ea

# 导入数据
data1 = pd.DataFrame(pd.read_excel('data/附件1.xlsx'))
data4 = pd.DataFrame(pd.read_excel('data/附件4.xlsx'))
data31 = pd.DataFrame(pd.read_excel('result/problem1/单日品类销售统计.xlsx'))
data32 = pd.DataFrame(pd.read_excel('result/problem2/品类单日利润率统计.xlsx'))
data33 = pd.DataFrame(pd.read_excel('data/部分进价统计.xlsx'))
data34 = pd.DataFrame(pd.read_excel('data/部分销售量统计.xlsx'))
data35 = pd.DataFrame(pd.read_excel('result/problem2/单品单日定价统计.xlsx'))

# 设置常量
cate_list = ['花叶类', '花菜类', '水生根茎类', '茄类', '辣椒类', '食用菌']
numbers = list(data33.iloc[:, 1])
L = []
n = 33 * 2
discount = 0.6


# 预测函数，使用随机森林算法
def predict(data, start, day):
    pred = []
    R2 = {}
    for i in range(data.shape[0]):
        cate = data.iloc[i, start - 1]
        sequence = data.iloc[i, start:]
        X = range(1, 1086)
        X_2d = [[x] for x in X]
        sequence = sequence.fillna(sequence.mean())
        sequence.replace([np.inf, -np.inf], 0, inplace=True)
        model = RandomForestRegressor(random_state=0, max_depth=5)
        model.fit(X_2d, sequence)
        pred.append(model.predict([[day]]))
        R2[cate] = model.score(X_2d, sequence)
    return [item for sublist in pred for item in sublist], R2


# 预测下面7天的销售量
sale_pred, sale_R2 = predict(data31, 2, 1086)

# 计算30日内的各品类销售量均值
sale_mean = []
shape = data34.shape[1]
for i in range(data34.shape[0]):
    mean = np.mean(data34.iloc[i, shape - 30: shape])
    sale_mean.append(mean)

# 计算各单品7日内定价均值
P_pred = []
shape = data35.shape[1]
for i in range(len(numbers)):
    for j in range(data35.shape[0]):
        if data35.iloc[j, 1] == numbers[i]:
            mean = np.mean(data35.iloc[j, shape - 7: shape])
            P_pred.append(mean)

# 预测下面7天的各单品进价
pred = []
for i in range(data33.shape[0]):
    sequence0 = data33.iloc[i, 1079:]
    X = range(1078, 1086)
    X_2d = [[x] for x in X]
    sequence0 = sequence0.fillna(sequence0.mean())
    sequence0.replace([np.inf, -np.inf], 0, inplace=True)
    model = RandomForestRegressor(random_state=0, max_depth=5)
    model.fit(X_2d, sequence0)
    pred.append(model.predict([[1086]]))
Q = [item for sublist in pred for item in sublist]

# 统计各单品损耗率
for i in range(n // 2):
    for j in range(data4.shape[0]):
        if data4.iloc[j, 0] == numbers[i]:
            L.append(data4.iloc[j, 2] / 100)


# 定义销售量变化与定价变动函数
def Atan(x):
    if x < 1.376:
        return math.tan(x) * (1 / math.pi)
    else:
        return -0.3


# 遗传算法
# 目标函数构建即约束不等式的构建
@ea.Problem.single
def evalVars(Vars):
    P = Vars[: n // 2]
    m = Vars[n // 2:]
    F = 0
    CV = []
    cv = 0
    for i in range(n // 2 + 1):
        if i < n // 2:
            N = (1 + Atan((P[i] - P_pred[i]) / P[i])) * sale_mean[i]
            if m[i] >= N:
                F += N * P[i] - Q[i] * m[i]
            else:
                F += m[i] * P[i] * (1 - L[i] + L[i] * discount) - Q[i] * m[i]

        if i <= 14:
            cv += -m[i]
        elif 14 < i <= 16:
            if i == 15:
                CV.append(cv + sale_pred[0])
                cv = 0
            cv += -m[i]
        elif 16 < i <= 19:
            if i == 17:
                CV.append(cv + sale_pred[1])
                cv = 0
            cv += -m[i]
        elif 19 < i <= 23:
            if i == 20:
                CV.append(cv + sale_pred[2])
                cv = 0
            cv += -m[i]
        elif 23 < i <= 28:
            if i == 24:
                CV.append(cv + sale_pred[3])
                cv = 0
            cv += -m[i]
        elif 28 < i <= 32:
            if i == 29:
                CV.append(cv + sale_pred[4])
                cv = 0
            cv += -m[i]
        else:
            CV.append(cv + sale_pred[5])
    return F, np.array(CV)


# 约束范围构建
types = [0] * (n // 2) + [0] * (n // 2)
lb = [0.01] * (n // 2) + [2.5] * (n // 2)
ub = [15] * (n // 2) + [150] * (n // 2)

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
                                 MAXGEN=100,
                                 logTras=1,
                                 trappedValue=1e-6,
                                 maxTrappedCount=10)

# 求解
res = ea.optimize(algorithm, seed=1, verbose=True, drawing=1, outputMsg=True, drawLog=False, saveFlag=True,
                  dirName='result/problem3/solver')
