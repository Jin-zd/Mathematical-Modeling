"""
problem1-2.py 使用多种群协同的增强精英保留的遗传算法确定2024-2025年除豆类外农作物的最佳种植方案。

主要功能包括：
- 使用`geatpy`库构建遗传算法
- 分别优化粮食、第一季蔬菜、萝卜、第二季蔬菜的种植方案
- 将结果导出到Excel文件
"""


from tools import *

# 2024,2025 年粮食种植方案，需要根据年份调整参数
I1 = range(0, 25)
J1 = range(0, 58)
K1 = range(5, 15)

dim_x, dim_y = len(J1), len(K1)
last_season_data = pd.read_excel('./results/1-2024/2024-粮食50.xlsx')
last_season_data = last_season_data.iloc[:, 2:].to_numpy()

@ea.Problem.single
def func1(x, Y=Y2):
    x = x.reshape(dim_x, dim_y)
    sum_y = 0
    for k in K1:
        sum_y += Y(x, k, J1)
    sum_jk = 0
    for j in J1:
        for k in K1:
            sum_jk += x[j_transform(j), k_transform(k)] * C(j, k) * S(j)
    f = sum_y - sum_jk

    cv = []
    for j in J1:
        j0 = F(j)
        if e[j0, 2] == 1:
            cv.append(np.sum(x[j_transform(j), :]))
        for k in K1:
            if x2023[j0, k] >= 1:
                cv.append(x[j_transform(j), k_transform(k)])
            if last_season_data[j_transform(j), k_transform(k)] == 1:
                cv.append(x[j_transform(j), k_transform(k)])
    for k in K1:
        sum_i = 0
        for i in I1:
            sum_i += M(i, k, x, J1)
        cv.append(sum_i - 4)

    sum_k = x.sum(axis=1)
    for x_s in sum_k:
        cv.append(x_s - 1)

    cv.append(800000 - f)
    return f, cv

# res1 = multi_sega(dim_x, dim_y, func1, prophet=None)
# result_to_excel(np.array(res1.get('Vars')).reshape(dim_x, dim_y), J1, K1, './results/1-2025/2025-粮食50.xlsx')


# 2024,2025 年蔬菜种植方案，需要根据年份调整参数
I2 = range(25, 53)
J2 = range(58, 106)
K2 = [15] + list(range(19, 34))

dim_x, dim_y = len(J2), len(K2)

@ea.Problem.single
def func2(x, Y=Y2):
    x = x.reshape(dim_x, dim_y)
    sum_y = 0
    for k in K2:
        sum_y += Y(x, k, J2)
    sum_jk = 0
    for j in J2:
        for k in K2:
            sum_jk += x[j_transform(j), k_transform(k)] * C(j, k) * S(j)
    f = sum_y - sum_jk

    cv = []
    for j in J2:
        j0 = F(j)
        if j0 == 32 or j0 == 33:
            if e[j0, 1] == 1:
                cv.append(np.sum(x[j_transform(j), :]))
        for k in K2:
            if x2023[j0, k] >= 1:
                cv.append(x[j_transform(j), k_transform(k)])
    for k in K2:
        sum_i = 0
        for i in I2:
            sum_i += M(i, k, x, J2)
        cv.append(sum_i - 4)

    sum_k = x.sum(axis=1)
    for x_s in sum_k:
        cv.append(x_s - 1)

    for j in J2:
        for k in K2:
            if D(j) not in B(k):
                cv.append(x[j_transform(j), k_transform(k)])
    cv.append(500000 - f)

    return f, cv

res2 = multi_sega(dim_x, dim_y, func2, prophet=None)
result_to_excel(np.array(res2.get('Vars')).reshape(dim_x, dim_y), J2, K2, './results/1-2024/2025-蔬菜50.xlsx')


# 2024,2025 年萝卜种植方案，需要根据年份调整参数
I3 = [27, 35]
J3 = range(58, 66)
K3 = range(34, 37)

last_season_data = pd.read_excel('./results/1-2025/2025-蔬菜50.xlsx')
last_season_data = last_season_data.iloc[:8, 2].to_numpy()
dim_x, dim_y = len(J3), len(K3)

@ea.Problem.single
def func3(x, Y=Y2):
    x = x.reshape(dim_x, dim_y)
    sum_y = 0
    for k in K3:
        sum_y += Y(x, k, J3)
    sum_jk = 0
    for j in J3:
        for k in K3:
            sum_jk += x[j_transform(j), k_transform(k)] * C(j, k) * S(j)
    f = sum_y - sum_jk

    cv = []
    for j in J3:
        j0 = F(j)
        if last_season_data[j0 - 26] == 1:
            cv.append(np.sum(x[j_transform(j), :]))

    for k in K3:
        sum_i = 0
        for i in I3:
            sum_i += M(i, k, x, J3)
        cv.append(sum_i - 4)

    sum_k = x.sum(axis=1)
    for x_s in sum_k:
        cv.append(x_s - 1)
    return f, cv

res3 = multi_sega(dim_x, dim_y, func3, prophet=None)
result_to_excel(np.array(res3.get('Vars')).reshape(dim_x, dim_y), J3, K3, './results/1-2025/2025-萝卜50-第二季.xlsx')

# 2024,2025 年蔬菜第二季种植方案，需要根据年份调整参数
last_season_data = pd.read_excel('./results/1-2025/2025-蔬菜50.xlsx')
last_season_data = last_season_data.iloc[40: 48].to_numpy()

I4 = range(50, 54)
J4 = range(98, 106)
K4 = range(19, 34)

dim_x, dim_y = len(J4), len(K4)

@ea.Problem.single
def func4(x, Y=Y1):
    x = x.reshape(dim_x, dim_y)
    sum_y = 0
    for k in K4:
        sum_y += Y(x, k, J4)
    sum_jk = 0
    for j in J4:
        for k in K4:
            sum_jk += x[j_transform(j), k_transform(k)] * C(j, k) * S(j)
    f = sum_y - sum_jk

    cv = []
    for j in J4:
        j0 = F(j)
        if e[j0, 1] == 1:
            cv.append(np.sum(x[j_transform(j), :]))
        for k in K4:
            if x2023[j0, k] >= 1:
                cv.append(x[j_transform(j), k_transform(k)])
            if last_season_data[j_transform(j), k_transform(k)] == 1:
                cv.append(x[j_transform(j), k_transform(k)])
    for k in K4:
        sum_i = 0
        for i in I4:
            sum_i += M(i, k, x, J4)
        cv.append(sum_i - 4)

    sum_k = x.sum(axis=1)
    for x_s in sum_k:
        cv.append(x_s - 1)

    for j in J4:
        for k in K4:
            if D(j) not in B(k):
                cv.append(x[j_transform(j), k_transform(k)])

    return f, cv

res4 = multi_sega(dim_x, dim_y, func4, prophet=None)
result_to_excel(np.array(res4.get('Vars')).reshape(dim_x, dim_y), J4, K4, './results/1-2025/2025-蔬菜50-第二季.xlsx')


