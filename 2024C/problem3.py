"""
problem3.py 将农作物进行分组，分别求解各个作物的种植方案

主要功能包括：
- 对农作物进行分组
- 构建遗传算法
- 求解各个作物的种植方案
- 将结果保存到excel文件中
"""


from problem3_data import *

# 粮食组
crop_bean = [0, 1, 2, 3 , 4]
crop_staple = [5, 6, 7, 8, 9, 14, 15]
crop_grain = [10, 13]
crop_tuber = [11, 12]

# 食用菌
mushroom = [37, 38, 39, 40]

# 蔬菜组
vage_bean = [16, 17, 18]
vage_egg = [20, 21, 23, 28, 30]
vage_leaf = [22, 26, 27, 29, 31, 32, 33, 34]
vage_cabbage = [24, 25]
vage_root = [19, 35, 36]

def Y(x, k, J, crop_land_to_yield_t, crop_price_t):
    return X(x, k, J, crop_land_to_yield_t) * P(k, crop_price_t)


last_season_data = pd.read_excel('./results/3-2029/2029-粮食.xlsx')
last_season_data = last_season_data.iloc[:, 2:].to_numpy()
I1 = range(0, 25)
J1 = range(0, 58)
K1 = range(5, 15)
dim_x, dim_y = len(J1), len(K1)

@ea.Problem.single
def func1(x, Y=Y):
    x = x.reshape(dim_x, dim_y)
    sum_y = 0
    for k in K1:
        sum_y += Y(x, k, J1, crop_land_to_yield_2030, crop_price_2030)
    sum_jk = 0
    for j in J1:
        for k in K1:
            sum_jk += x[j_transform(j), k_transform(k)] * C(j, k, crop_land_to_cost_2030) * S(j)
    f = sum_y - sum_jk

    cv = []
    for j in J1:
        j0 = F(j)
        if e[j0, 1] == 1:
            cv.append(np.sum(x[j, :]))
        for k in K1:
            if last_season_data[j_transform(j), k_transform(k)] == 1:
                cv.append(x[j_transform(j), k_transform(k)])
    for k in K1:
        sum_i = 0
        for i in I1:
            sum_i += M(i, k, x, J1)
        cv.append(sum_i - 4)

    for crops in [crop_staple, crop_grain, crop_tuber]:
        sum_land_c = 0
        sum_land_r = 0
        for k in crops:
            if k != 15:
                sum_land_c += X(x, k, J1, crop_land_to_yield_2030)
                sum_land_r += R(k, crop_sale_2030)
        cv.append(sum_land_c - sum_land_r)

    sum_k = x.sum(axis=1)
    for x_s in sum_k:
        cv.append(x_s - 1)
    cv.append(500000 - f)
    return f, cv

# res1 = multi_sega(dim_x, dim_y, func1, prophet=None)
# result_to_excel(np.array(res1.get('Vars')).reshape(dim_x, dim_y), J1, K1, './results/3-2030/2030-粮食.xlsx')


last_season_data = pd.read_excel('./results/3-2029/2029-蔬菜.xlsx')
last_season_data = last_season_data.iloc[:, 2:].to_numpy()
I2 = range(25, 53)
J2 = range(58, 106)
K2 = [15] + list(range(19, 34))

dim_x, dim_y = len(J2), len(K2)

@ea.Problem.single
def func2(x, Y=Y):
    x = x.reshape(dim_x, dim_y)
    sum_y = 0
    for k in K2:
        sum_y += Y(x, k, J2, crop_land_to_yield_2030, crop_price_2030)
    sum_jk = 0
    for j in J2:
        for k in K2:
            sum_jk += x[j_transform(j), k_transform(k)] * C(j, k, crop_land_to_cost_2030) * S(j)
    f = sum_y - sum_jk

    cv = []
    for j in J2:
        j0 = F(j)
        if j0 == 32 or j0 == 33:
            if e[j0, 1] == 1:
                cv.append(np.sum(x[j_transform(j), :]))
        for k in K2:
            if k == 15:
                if last_season_data[j_transform(j), k_transform(k)] == 1:
                    cv.append(x[j_transform(j), k_transform(k)])

    for k in K2:
        sum_i = 0
        for i in I2:
            sum_i += M(i, k, x, J2)
        cv.append(sum_i - 4)

    for crops in [vage_egg, vage_leaf, vage_cabbage, vage_root]:
        sum_land_c = 0
        sum_land_r = 0
        for k in crops:
            if k in K2:
                sum_land_c += X(x, k, J2, crop_land_to_yield_2030)
                sum_land_r += R(k, crop_sale_2030)
        cv.append(sum_land_c - sum_land_r)

    sum_k = x.sum(axis=1)
    for x_s in sum_k:
        cv.append(x_s - 1)

    for j in J2:
        for k in K2:
            if D(j) not in B(k):
                cv.append(x[j_transform(j), k_transform(k)])
    cv.append(560000 - f)

    return f, cv

# res2 = multi_sega(dim_x, dim_y, func2, prophet=None)
# result_to_excel(np.array(res2.get('Vars')).reshape(dim_x, dim_y), J2, K2, './results/3-2030/2030-蔬菜.xlsx')


I3 = [27, 35]
J3 = range(58, 66)
K3 = range(34, 37)

last_season_data = pd.read_excel('./results/3-2030/2030-蔬菜.xlsx')
last_season_data = last_season_data.iloc[:8, 2].to_numpy()
dim_x, dim_y = len(J3), len(K3)

@ea.Problem.single
def func3(x, Y=Y):
    x = x.reshape(dim_x, dim_y)
    sum_y = 0
    for k in K3:
        sum_y += Y(x, k, J3, crop_land_to_yield_2030, crop_price_2030)
    sum_jk = 0
    for j in J3:
        for k in K3:
            sum_jk += x[j_transform(j), k_transform(k)] * C(j, k, crop_land_to_cost_2030) * S(j)
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

    for crops in [vage_leaf, vage_root]:
        sum_land_c = 0
        sum_land_r = 0
        for k in crops:
            if k in K3:
                sum_land_c += X(x, k, J3, crop_land_to_yield_2030)
                sum_land_r += R(k, crop_sale_2030)
        cv.append(sum_land_c - sum_land_r)

    sum_k = x.sum(axis=1)
    for x_s in sum_k:
        cv.append(x_s - 1)
    return f, cv

# res3 = multi_sega(dim_x, dim_y, func3, prophet=None)
# result_to_excel(np.array(res3.get('Vars')).reshape(dim_x, dim_y), J3, K3, './results/3-2030/2030-萝卜-第二季.xlsx')

last_season_data = pd.read_excel('./results/3-2029/2029-蔬菜.xlsx')
last_season_data = last_season_data.iloc[40: 48].to_numpy()

I4 = range(50, 54)
J4 = range(98, 106)
K4 = range(19, 34)

dim_x, dim_y = len(J4), len(K4)

@ea.Problem.single
def func4(x, Y=Y):
    x = x.reshape(dim_x, dim_y)
    sum_y = 0
    for k in K4:
        sum_y += Y(x, k, J4, crop_land_to_yield_2030, crop_price_2030)
    sum_jk = 0
    for j in J4:
        for k in K4:
            sum_jk += x[j_transform(j), k_transform(k)] * C(j, k, crop_land_to_cost_2030) * S(j)
    f = sum_y - sum_jk

    cv = []
    for j in J4:
        j0 = F(j)
        if e[j0, 0] == 1:
            cv.append(np.sum(x[j_transform(j), :]))
        for k in K4:
            if last_season_data[j_transform(j), k_transform(k)] == 1:
                cv.append(x[j_transform(j), k_transform(k)])
    for k in K4:
        sum_i = 0
        for i in I4:
            sum_i += M(i, k, x, J4)
        cv.append(sum_i - 4)


    for crops in [vage_egg, vage_leaf, vage_cabbage, vage_root]:
        sum_land_c = 0
        sum_land_r = 0
        for k in crops:
            if k in K4:
                sum_land_c += X(x, k, J4, crop_land_to_yield_2030)
                sum_land_r += R(k, crop_sale_2030)
        cv.append(sum_land_c - sum_land_r)

    sum_k = x.sum(axis=1)
    for x_s in sum_k:
        cv.append(x_s - 1)

    for j in J4:
        for k in K4:
            if D(j) not in B(k):
                cv.append(x[j_transform(j), k_transform(k)])

    return f, cv

# res4 = multi_sega(dim_x, dim_y, func4, prophet=None)
# result_to_excel(np.array(res4.get('Vars')).reshape(dim_x, dim_y), J4, K4, './results/3-2030/2030-蔬菜-第二季.xlsx')

I5 = range(34, 50)
J5 = range(66, 98)
K5 = range(37, 41)

dim_x, dim_y = len(J5), len(K5)

@ea.Problem.single
def func5(x, Y=Y):
    x = x.reshape(dim_x, dim_y)
    sum_y = 0
    for k in K5:
        sum_y += Y(x, k, J5, crop_land_to_yield_2030, crop_price_2030)
    sum_jk = 0
    for j in J5:
        for k in K5:
            sum_jk += x[j_transform(j), k_transform(k)] * C(j, k, crop_land_to_cost_2030) * S(j)
    f = sum_y - sum_jk

    cv = []
    for k in K5:
        sum_i = 0
        for i in I5:
            sum_i += M(i, k, x, J5)
        cv.append(sum_i - 4)

    sum_land_c = 0
    sum_land_r = 0
    for k in mushroom:
        sum_land_c += X(x, k, J5, crop_land_to_yield_2030)
        sum_land_r += R(k, crop_sale_2030)
    cv.append(sum_land_c - sum_land_r)

    sum_k = x.sum(axis=1)
    for x_s in sum_k:
        cv.append(x_s - 1)

    return f, cv

res5 = multi_sega(dim_x, dim_y, func5, prophet=None)
result_to_excel(np.array(res5.get('Vars')).reshape(dim_x, dim_y), J5, K5, './results/3-2030/2030-食用菌-第二季.xlsx')