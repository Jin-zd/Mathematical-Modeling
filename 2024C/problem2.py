"""
problem2.py 实现了2024-2030的农作物种植的随机规划。
主要功能包括：
- 导入随机数据
- 使用`geatpy`库实现遗传算法
- 优化各年各类别的种植方案
"""
import matplotlib.pyplot as plt

from problem2_data import *

# 单一目标函数
def target(x, Y, J, K, crop_sale_t, crop_land_to_yield_t, crop_price_t, crop_land_to_cost_t):
    x = x.reshape(dim_x, dim_y)
    sum_y = 0
    for k in K:
        sum_y += Y(x, k, J, crop_sale_t, crop_land_to_yield_t, crop_price_t)
    sum_jk = 0
    for j in J:
        for k in K:
            sum_jk += x[j_transform(j), k_transform(k)] * C(j, k, crop_land_to_cost_t) * S(j)
    return sum_y - sum_jk

last_season_data = pd.read_excel('./results/2-2024/2024-粮食.xlsx')
last_season_data = last_season_data.iloc[:, 2:].to_numpy()

I1 = range(0, 25)
J1 = range(0, 58)
K1 = range(5, 15)
dim_x, dim_y = len(J1), len(K1)

@ea.Problem.single
def func1(x, Y=Y2):
    x = x.reshape(dim_x, dim_y)
    f = 0
    for n in range(n_handle):
        f += target(x, Y, J1, K1, crop_sale_list_2026[n], crop_land_to_yield_list_2026[n], crop_price_list_2026[n],
                    crop_land_to_cost_copy_2026)
    f = f / n_handle # 优化多次的目标函数总和

    cv = []
    for j in J1:
        j0 = F(j)
        if e[j0, 1] == 1:
            cv.append(np.sum(x[j, :]))
        for k in K1:
            # if x2023[j0, k] >= 1:
            #     cv.append(x[j_transform(j), k_transform(k)])
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

    cv.append(400000 - f)
    return f, cv

res1 = multi_sega(dim_x, dim_y, func1, prophet=None)
result_to_excel(np.array(res1.get('Vars')).reshape(dim_x, dim_y), J1, K1, './results/2-2025/2025-粮食.xlsx')

# 鲁棒性检验
profit_test = []
for i in range(20):
    res = multi_sega(dim_x, dim_y, func1, prophet=None, random_seed=np.random.randint(0, 1000))
    profit_test.append(res.get('ObjV')[-1])

plt.rc('font', family='SimHei')
plt.rc('axes', unicode_minus=False)
mean = np.mean(profit_test)
fig, ax = plt.subplots(figsize=(12, 6))
upper_bound = mean * 1.02
lower_bound = mean * 0.98
ax.fill_between(np.arange(len(profit_test)), lower_bound, upper_bound,
                alpha=0.2, color='#55A868', label='均值±2%范围')
ax.plot(profit_test, color='#4C72B0', linewidth=2, label='利润')
ax.axhline(y=upper_bound, color='#8172B3', linestyle=':', linewidth=1.5, label='上限 (均值+2%)')
ax.axhline(y=lower_bound, color='#8172B3', linestyle=':', linewidth=1.5, label='下限 (均值-2%)')
ax.set_ylim([4500000, 5500000])
ax.set_xlabel('运行次数', fontsize=12)
ax.set_ylabel('目标函数值', fontsize=12)
ax.set_title('利润鲁棒性检验', fontsize=16, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, linestyle=':', alpha=0.7)
ax.set_xticks(range(0, 20, 2))
ax.set_xticklabels(range(1, 21, 2))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.tight_layout()
plt.savefig('./pictures/鲁棒性检验.png', dpi=300, bbox_inches='tight')
plt.close()


last_season_data = pd.read_excel('./results/2-2029/2029-蔬菜.xlsx')
last_season_data = last_season_data.iloc[:, 2:].to_numpy()
I2 = range(25, 53)
J2 = range(58, 106)
K2 = [15] + list(range(19, 34))

dim_x, dim_y = len(J2), len(K2)

@ea.Problem.single
def func2(x, Y=Y2):
    x = x.reshape(dim_x, dim_y)
    f = 0
    for n in range(n_handle):
        f += target(x, Y, J2, K2, crop_sale_list_2030[n], crop_land_to_yield_list_2030[n], crop_price_list_2030[n],
                    crop_land_to_cost_copy_2030)
    f = f / n_handle

    cv = []
    for j in J2:
        j0 = F(j)
        if j0 == 32 or j0 == 33:
            if e[j0, 1] == 1:
                cv.append(np.sum(x[j_transform(j), :]))
        for k in K2:
        #     if x2023[j0, k] >= 1:
        #         cv.append(x[j_transform(j), k_transform(k)])
            if k == 15:
                if last_season_data[j_transform(j), k_transform(k)] == 1:
                    cv.append(x[j_transform(j), k_transform(k)])


    sum_k = x.sum(axis=1)
    for x_s in sum_k:
        cv.append(x_s - 1)

    for k in K2:
        sum_i = 0
        for i in I2:
            sum_i += M(i, k, x, J2)
        cv.append(sum_i - 4)

    for j in J2:
        for k in K2:
            if D(j) not in B(k):
                cv.append(x[j_transform(j), k_transform(k)])
    return f, cv

res2 = multi_sega(dim_x, dim_y, func2, prophet=None)
result_to_excel(np.array(res2.get('Vars')).reshape(dim_x, dim_y), J2, K2, './results/2-2030/2030-蔬菜.xlsx')

I3 = [27, 35]
J3 = range(58, 66)
K3 = range(34, 37)

last_season_data = pd.read_excel('./results/2-2024/2024-蔬菜.xlsx')
last_season_data = last_season_data.iloc[:8, 2].to_numpy()
dim_x, dim_y = len(J3), len(K3)

@ea.Problem.single
def func3(x, Y=Y2):
    x = x.reshape(dim_x, dim_y)
    f = 0
    for n in range(n_handle):
        f += target(x, Y, J3, K3, crop_sale_list_2030[n], crop_land_to_yield_list_2030[n], crop_price_list_2030[n],
                    crop_land_to_cost_copy_2030)
    f = f / n_handle

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
result_to_excel(np.array(res3.get('Vars')).reshape(dim_x, dim_y), J3, K3, './results/2-2024/2024-萝卜-第二季.xlsx')

last_season_data = pd.read_excel('./results/2-2024/2024-蔬菜.xlsx')
last_season_data = last_season_data.iloc[40: 48].to_numpy()

I4 = range(50, 54)
J4 = range(98, 106)
K4 = range(19, 34)

dim_x, dim_y = len(J4), len(K4)

@ea.Problem.single
def func4(x, Y=Y2):
    x = x.reshape(dim_x, dim_y)
    f = 0
    for n in range(n_handle):
        f += target(x, Y, J4, K4, crop_sale_list_2030[n], crop_land_to_yield_list_2030[n], crop_price_list_2030[n],
                    crop_land_to_cost_copy_2030)
    f = f / n_handle

    cv = []
    for j in J4:
        j0 = F(j)
        if e[j0, 2] == 1:
            cv.append(np.sum(x[j_transform(j), :]))
        for k in K4:
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
result_to_excel(np.array(res4.get('Vars')).reshape(dim_x, dim_y), J4, K4, './results/2-2030/2030-蔬菜-第二季.xlsx')


I5 = range(34, 50)
J5 = range(66, 98)
K5 = range(37, 41)

dim_x, dim_y = len(J4), len(K4)

@ea.Problem.single
def func5(x, Y=Y2):
    x = x.reshape(dim_x, dim_y)
    f = 0
    for n in range(n_handle):
        f += target(x, Y, J5, K5, crop_sale_list_2024[n], crop_land_to_yield_list_2024[n], crop_price_list_2024[n],
                    crop_land_to_cost_copy_2024)
    f = f / n_handle

    cv = []
    for k in K5:
        sum_i = 0
        for i in I5:
            sum_i += M(i, k, x, J5)
        cv.append(sum_i - 4)

    sum_k = x.sum(axis=1)
    for x_s in sum_k:
        cv.append(x_s - 1)

    return f, cv

res5 = multi_sega(dim_x, dim_y, func5, prophet=None)
result_to_excel(np.array(res5.get('Vars')).reshape(dim_x, dim_y), J5, K5, './results/2-2024/2024-食用菌-第二季.xlsx')




