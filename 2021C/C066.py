import numpy as np
import pandas as pd
import math
from scipy.optimize import minimize
import geatpy as ea

data1 = pd.DataFrame(pd.read_excel('附件1 近5年402家供应商的相关数据.xlsx', sheet_name=0))
data2 = pd.DataFrame(pd.read_excel('附件1 近5年402家供应商的相关数据.xlsx', sheet_name=1))

data2A = data2.loc[data2['材料分类'] == 'A'].copy()
data2B = data2.loc[data2['材料分类'] == 'B'].copy()
data2C = data2.loc[data2['材料分类'] == 'C'].copy()

data2A.to_excel('供货A.xlsx')
data2B.to_excel('供货B.xlsx')
data2C.to_excel('供货C.xlsx')

dic = data2.iloc[:, [0, 1]].copy()
data1.drop(data1.columns[0: 2], axis=1, inplace=True)
data2.drop(data2.columns[0: 2], axis=1, inplace=True)
data2A.drop(data2A.columns[0: 3], axis=1, inplace=True)
data2B.drop(data2B.columns[0: 3], axis=1, inplace=True)
data2C.drop(data2C.columns[0: 3], axis=1, inplace=True)

# 问题一求解
# 供货次数
del_num = []

for i in range(data2.shape[0]):
    count_sum = 0
    for j in range(data2.shape[1]):
        if data2.iloc[i, j] != 0:
            count_sum += 1
    del_num.append(count_sum)


# 平均供货量
def average_calculate(data):
    average = []
    for i in range(data.shape[0]):
        flag = dic.iloc[i][1]
        if flag == 'A':
            p = 0.6
        elif flag == 'B':
            p = 0.66
        else:
            p = 0.72
        average.append(data.sum(axis=1)[i] / (p * del_num[i]))
    return average


aver = average_calculate(data2)

# 单次最大供货量
x_max = []
x_maxSet = data2.max(axis=1)

for i in range(data2.shape[0]):
    x_max.append(x_maxSet[i])

# 供货稳定性
stab = []

for i in range(data2.shape[0]):
    stab_sum = 0
    for j in range(data2.shape[1]):
        stab_sum += (data2.iloc[i][j] - data1.iloc[i][j]) ** 2
    stab.append(stab_sum / del_num[i])

# 合理供货比例
del_rate = []

for i in range(data2.shape[0]):
    comp1 = data1.sum(axis=1)[i]
    comp2 = data2.sum(axis=1)[i]
    if 0.8 * comp1 < comp2 < 1.2 * comp1:
        del_rate.append(1 / del_num[i])
    else:
        del_rate.append(0)


# 极小型指标转化为极大型指标
def min_to_max(a):
    for i in range(len(a)):
        a[i] = 1 / a[i]


# 居中型指标转化为极大型指标
def center_to_max(a):
    a_max = max(a)
    a_min = min(a)
    for i in range(0, len(a)):
        if a_min <= a[i] < (a_min + a_max) / 2:
            a[i] = (a[i] - a_min) / (a_max - a_min)
        else:
            a[i] = (a_max - a[i]) / (a_max - a_min)


# 区间型指标转化为极大型指标
def inter_to_max(a, b1, b2):
    a_max = max(a)
    a_min = min(a)
    c = max(b1 - a_min, a_max - b2)
    for i in range(len(a)):
        if a[i] < b1:
            a[i] = 1 - (b1 - a[i]) / c
        elif b1 < a[i] < b2:
            a[i] = 1
        else:
            a[i] = 1 - (a[i] - b2) / c


# 标准样本变换法
def standard_transform(a):
    average = sum(a) / len(a)
    s = 0
    for i in range(len(a)):
        s += (a[i] - average) ** 2
    s = (s / (len(a) - 1)) ** (1 / 2)
    for i in range(len(a)):
        a[i] = (a[i] - average) / s


# TOPSIS评价函数
def topsis(*indicators):
    for indicator in indicators:
        center_to_max(indicator)
        standard_transform(indicator)
    c1 = []
    c2 = []
    for indicator in indicators:
        c1.append(max(indicator))
        c2.append(min(indicator))
    s1 = []
    s2 = []
    for j in range(len(indicators[0])):
        dist1 = 0
        dist2 = 0
        for i in range(len(indicators)):
            dist1 += (indicators[i][j] - c1[i]) ** 2
            dist2 += (indicators[i][j] - c2[i]) ** 2
        s1.append(dist1 ** (1 / 2))
        s2.append(dist2 ** (1 / 2))
    f = []
    for i in range(len(s1)):
        f.append(s2[i] / (s1[i] + s2[i]))
    return f


# 熵权法
def ewm(*indicators):
    p = []
    for i in range(len(indicators)):
        temp = []
        row_sum = sum(indicators[i])
        for j in range(len(indicators[i])):
            temp.append(indicators[i][j] / row_sum)
        p.append(temp)
    g = []
    for i in range(len(indicators)):
        e_sum = 0
        for j in range(len(indicators[i])):
            e_sum += p[i][j] * math.log(p[i][j])
        g.append(1 - 1 / math.log(len(indicators[i])) * e_sum)
    w = []
    g_sum = sum(g)
    for i in range(len(indicators)):
        w.append(g[i] / g_sum)
    return w


result1 = {}
f = topsis(del_num, aver, x_max, stab, del_rate)
for i in range(data2.shape[0]):
    result1['S' + str(i).zfill(3)] = f[i]

result1 = sorted(result1.items(), key=lambda x: x[1], reverse=True)
keys = []
values = []
for i in range(50):
    keys.append(result1[i][0])
    values.append(result1[i][1])

result1 = pd.DataFrame(columns=['供应商', '评价值'])
for i in range(50):
    result1.loc[len(result1.index)] = [keys[i], values[i]]
result1.to_excel('50家最重要的供货商.xlsx')

# 问题二求解
data3 = pd.DataFrame(pd.read_excel('附件2 近5年8家转运商的相关数据.xlsx'))

# 平均损失率
loss_rate = 0
for j in range(1, data3.shape[1]):
    week_sum = 0
    n_t = 0
    for i in range(1, data3.shape[0]):
        item = data3.iloc[i][j]
        if item != 0:
            n_t += 1
        week_sum += item
    loss_rate += week_sum / n_t
loss_rate = loss_rate / data3.shape[1]


# scipy库优化
def obj(y):
    exp1 = 0
    exp2 = 0
    for i in range(50):
        exp1 += y[i]
        exp2 += y[i] * values[i]
    if exp2 == 0 or np.isnan(exp2) or np.isinf(exp2):
        return np.inf
    return exp1 / exp2


def constr(y):
    exp = 0
    for i in range(50):
        exp += y[i] * aver[int(keys[i][3], 10)]
    exp = exp * loss_rate
    return exp - 2.82e4


con = {'type': 'ineq', 'fun': constr}
bd = tuple([(0, 1)] * 50)
res = minimize(obj, np.random.randn(50), constraints=con, bounds=bd)
print(res)


# geatpy库优化（遗传算法）
@ea.Problem.single
def eval_vars(y):
    f = obj(y)
    cv = np.array([-constr(y)])
    return f, cv


problem1 = ea.Problem(name='problem1',
                      M=1, maxormins=1,
                      Dim=50,
                      varTypes=[1 for _ in range(50)],
                      lb=[0 for _ in range(50)],
                      ub=[1 for _ in range(50)],
                      evalVars=eval_vars)
algorithm1 = ea.soea_SEGA_templet(problem1,
                                  ea.Population(Encoding='RI', NIND=100),
                                  MAXGEN=2000,
                                  logTras=1,
                                  trappedValue=1e-6,
                                  maxTrappedCount=10)
res1 = ea.optimize(algorithm1, seed=1, verbose=True, drawing=1, outputMsg=True, drawLog=False, saveFlag=True,
                   dirName='result1')
