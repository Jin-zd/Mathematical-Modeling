import re
import numpy as np
import pandas as pd
import geatpy as ea
from geatpy import Population

land_to_area1 = pd.read_excel('./new_data/data1.xlsx')
land_to_area2 = pd.read_excel('./new_data/data5.xlsx').iloc[:, :4].ffill()
land_to_area2 = land_to_area2.iloc[:, :4].ffill()
crop_to_land  = pd.read_excel('./new_data/data2.xlsx').iloc[:42, :]
history_crop_to_land = pd.read_excel('./new_data/data3.xlsx')
crop_to_sale = pd.read_excel('./new_data/data4.xlsx')
grain_bean_2023 = pd.read_excel('./results/豆类/2023-豆类1.xlsx')
grain_bean_2024 = pd.read_excel('./results/豆类/2024-豆类1.xlsx')
grain_bean_2025 = pd.read_excel('./results/豆类/2025-豆类1.xlsx')
vge_bean_2023 = pd.read_excel('./results/豆类/2023-豆类2.xlsx')
vge_bean_2024 = pd.read_excel('./results/豆类/2024-豆类2.xlsx')
vge_bean_2025 = pd.read_excel('./results/豆类/2025-豆类2.xlsx')

land_to_area1['地块类型'] = land_to_area1['地块类型'].str.strip()
land_to_area2['地块类型'] = land_to_area2['地块类型'].str.strip()
crop_to_land['作物名称'] = crop_to_land['作物名称'].str.strip()
crop_to_sale['作物名称'] = crop_to_sale['作物名称'].str.strip()
crop_to_sale['地块类型'] = crop_to_sale['地块类型'].str.strip()
history_crop_to_land['作物名称'] = history_crop_to_land['作物名称'].str.strip()

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text for text in re.split('(\d+)', s)]

history_crop_to_land = history_crop_to_land.sort_values(by='地块名称', key=lambda col: col.map(natural_sort_key))

# 地块-索引 映射
land_index = {value: index for index, value in enumerate(land_to_area1.iloc[:, 0].tolist())}

# 索引-地块 映射
index_land = land_to_area1.iloc[:, 0].tolist()

# 基本地块-地块 映射
basic_land_list = land_to_area2.iloc[:, 0].to_list()

# 基本地块-面积 映射
basic_area_list = land_to_area2.iloc[:, 3].to_list()
keys = land_to_area2.iloc[:, 0].unique().tolist()
land_dict = {}
for key in keys:
    land_dict[key] = land_to_area2[land_to_area2.iloc[:, 0] == key].iloc[:, 1].to_list()
# 地块-基本地块 映射
land_dict = list(land_dict.values())

# 作物-索引 映射
crop_index = {value: index for index, value in enumerate(crop_to_land.iloc[:, 0].tolist())}

# 索引-作物 映射
index_crop = crop_to_land.iloc[:, 0].tolist()

# 作物-单价 映射
crop_price = crop_to_sale.set_index('作物名称')['销售单价'].to_dict()


# 地块编号-地块类型 映射
land_type = land_to_area1.iloc[:, 1].to_list()


# 地块类型^作物->种植成本 映射
crop_land_to_cost = crop_to_sale.groupby('地块类型').apply(
    lambda g: dict(zip(g['作物名称'], g['种植成本']))).to_dict()


# 地块类型^作物->亩产量 映射
crop_land_to_yield = crop_to_sale.groupby('地块类型').apply(
    lambda g: dict(zip(g['作物名称'], g['亩产量']))).to_dict()


# 作物-作物类型 映射
crop_type = crop_to_land.set_index('作物名称')['作物类型'].to_dict()

# 作物-销售量 映射
crop_sale = {}
crop_land_groups = history_crop_to_land.groupby('作物名称')
crop_product_groups = crop_to_sale.groupby('作物名称')
for (name1, land_group), (name2, product_group) in zip(crop_land_groups, crop_product_groups):
    key = land_group['作物名称'].tolist()[0]
    product_sum_outer = 0
    for row in land_group.iterrows():
        land_area = row[1]['作物种植面积']
        land_name = row[1]['地块名称']
        land_index_c = land_index[land_name]
        land_type_c = land_type[land_index_c]
        row_product = product_group[product_group['地块类型'] == land_type_c]
        land_product = row_product['亩产量'].values[0]
        product_sum = land_product * land_area
        product_sum_outer += product_sum
    crop_sale[key] = product_sum_outer

# 作物-要求耕地 映射
crop_to_land_require = {
    row['作物名称']: [s for s in row['种植耕地和季次'].split('\n') if s]
    for _, row in crop_to_land.iterrows()
}

x2023 = np.zeros((86, 42))
for index_now in range(history_crop_to_land.shape[0]):
    land_name = history_crop_to_land.iloc[index_now, 0]
    crop_name = history_crop_to_land.iloc[index_now, 1]
    crop_index_now = crop_index[crop_name]
    land_index_now = land_index[land_name]
    x2023[land_index_now, crop_index_now] = 1
pd.DataFrame(x2023).to_excel('./results/1-6.xlsx', index=False)

# 基本地块编号->地块编号
def F(k: int) -> int:
    return land_index[basic_land_list[k]]


# 基本地块编号->面积
def S(j: int) -> int:
    return basic_area_list[j]


# 地块编号->基本地块编号
def G(i: int)-> list[int]:
    return land_dict[i]


# 作物编号->单价
def P(k: int, crop_price_t)-> float:
    return crop_price_t[index_crop[k]]


# 地块编号^作物编号->种植成本
def C(j: int, k: int, crop_land_to_cost_t=None)-> float:
    if crop_land_to_cost_t is None:
        crop_land_to_cost_t = crop_land_to_cost

    i = F(j)
    m = land_type[i]
    if m not in B(k):
        return 100000
    else:
        return crop_land_to_cost_t[m][index_crop[k]]

# 地块编号^作物编号->亩产量
def Q(j: int, k: int, crop_land_to_yield_t):
    i = F(j)
    m = land_type[i]
    if m not in B(k):
        return 0
    else:
        return crop_land_to_yield_t[m][index_crop[k]]


# 作物编号->销售量
def R(k: int, crop_sale_t):
    return crop_sale_t[index_crop[k]]


# 基本地块编号->地块类型
def D(j: int):
    i = F(j)
    return land_type[i]


def e_generate(grain_data, vge_data):
    e = np.zeros((54, 1))
    mask1 = (grain_data.iloc[:, 1:].sum(axis=1) > 0).values
    mask2 = (vge_data.iloc[:, 1:].sum(axis=1) > 0).values
    e[:26, 0] = mask1
    e[26:, 0] = mask2
    return e

e = np.hstack(
    [e_generate(grain_bean_2023, vge_bean_2023),
     e_generate(grain_bean_2024, vge_bean_2024),
     e_generate(grain_bean_2025, vge_bean_2025)]
)


# 基本地块编号^年份编号->地块是否种植豆类
def E(j: int, t: int)-> bool:
    i = F(j)
    return e[i, t]


# 作物编号->作物要求耕地
def B(k: int)-> list[str]:
    return crop_to_land_require[index_crop[k]]


def X(x, k, J, crop_land_to_yield_t):
    sum_x = 0
    for j in J:
        sum_x += x[j_transform(j), k_transform(k)] * S(j) * Q(j, k, crop_land_to_yield_t)
    return sum_x


def Y1(x, k, J, crop_sale_t=None, crop_land_to_yield_t=None, crop_price_t=None):
    if crop_sale_t is None:
        crop_sale_t = crop_sale
    if crop_land_to_yield_t is None:
        crop_land_to_yield_t = crop_land_to_yield
    if crop_price_t is None:
        crop_price_t = crop_price

    if X(x, k, J, crop_land_to_yield_t) <= R(k, crop_sale_t):
        return X(x, k, J, crop_land_to_yield_t) * P(k, crop_price_t)
    else:
        return R(k, crop_sale_t) * P(k, crop_price_t)


def Y2(x, k, J, crop_sale_t=None, crop_land_to_yield_t=None, crop_price_t=None):
    if crop_sale_t is None:
        crop_sale_t = crop_sale
    if crop_land_to_yield_t is None:
        crop_land_to_yield_t = crop_land_to_yield
    if crop_price_t is None:
        crop_price_t = crop_price

    if X(x, k, J, crop_land_to_yield_t) <= R(k, crop_sale_t):
        return X(x, k, J, crop_land_to_yield_t) * P(k, crop_price_t)
    else:
        return R(k, crop_sale_t) * P(k, crop_price_t) + (X(x, k, J, crop_land_to_yield_t) - R(k, crop_sale_t)) * P(k, crop_price_t) * 0.5


def M(i, k, x, J):
    sum_j = 0
    for j in J:
        if j in G(i):
            sum_j += x[j_transform(j), k_transform(k)]
    return int(sum_j != 0)


def k_transform(k):
    # if k == 15:
    #     return 0
    # else:
    #     return k - 18
    return k - 37

def j_transform(j):
    return j - 66


def result_to_excel(value, J, K, file_name):
    crops = [index_crop[k] for k in K]
    data = {
        '地块名称': [index_land[F(j)] for j in J],
        '基本地块': [j + 1 for j in J],
    }
    for i, crop in enumerate(crops):
        data[crop] = value[:, i]
    pd.DataFrame(data).to_excel(file_name, index=False)


def multi_sega(dim_x, dim_y, func, prophet=None):
    dim = dim_x * dim_y
    ub = [1] * dim
    lb = [0] * dim

    problem = ea.Problem(
        name='problem',
        M=1,
        maxormins=[-1],
        Dim=dim,
        varTypes=[1] * dim,
        lb=lb,
        ub=ub,
        evalVars=func)

    field1 = ea.crtfld('BG', problem.varTypes, problem.ranges)
    field2 = ea.crtfld('BG', problem.varTypes, problem.ranges)
    field3 = ea.crtfld('BG', problem.varTypes, problem.ranges)
    population = [
        Population(Encoding='BG', Field=field1, NIND=20),
        Population(Encoding='BG', Field=field2, NIND=20),
        Population(Encoding='BG', Field=field3, NIND=20)
    ]

    algorithm = ea.soea_multi_SEGA_templet(
        problem,
        population,
        MAXGEN=3000,
        logTras=1,
        trappedValue=1e-7,
        maxTrappedCount=10)

    result = ea.optimize(algorithm,
                      seed=np.random.randint(0, 1000),
                      prophet=prophet,
                      verbose=False,
                      drawing=0,
                      outputMsg=True,
                      drawLog=False,
                      saveFlag=False)
    return result
