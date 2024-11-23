"""
problem1-1.py 使用多种群协同的遗传算法确定2024-2025年豆类（粮食豆类和蔬菜豆类）的最佳种植方案。

主要功能包括：
- 数据导入和预处理
- 使用`geatpy`库实现遗传算法
- 优化粮食豆类和蔬菜豆类的种植方案
- 将结果导出到Excel文件
"""


import re
import numpy as np
import pandas as pd
import geatpy as ea
from geatpy import Population


# 数据导入
land_to_area1 = pd.read_excel('./new_data/data1.xlsx')
land_to_area2 = pd.read_excel('./new_data/data5.xlsx')
land_to_area2 = land_to_area2.iloc[:, :4].ffill()
crop_to_land  = pd.read_excel('./new_data/data2.xlsx')
crop_to_sale = pd.read_excel('./new_data/data4.xlsx')
history_crop_to_land = pd.read_excel('./new_data/data3.xlsx')

# 数据预处理，生成算法所需的数据
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text for text in re.split('(\d+)', s)]

history_crop_to_land = history_crop_to_land.sort_values(by='地块名称', key=lambda col: col.map(natural_sort_key))


# 遗传算法：求解2024-2025豆类种植方案
# 处理粮食豆类
grain_bean = ['黄豆', '黑豆', '红豆', '绿豆', '爬豆']
grain_bean_dict = {key: value for key, value in zip(grain_bean, range(5))}
x0 = np.zeros((26, 5))
for i in range(26):
    grain_name = history_crop_to_land.loc[i, '作物名称']
    if grain_name in grain_bean_dict:
        x0[i, grain_bean_dict[grain_name]] = 1
    else:
        continue

grain_s = land_to_area1['地块面积'].to_numpy()[:26].reshape(-1, 1)

dim = 26 * 5 * 2

@ea.Problem.single
def func1(x):
    x1 = x[:130].reshape(26, 5)
    x2 = x[130:].reshape(26, 5)
    f = np.abs((x1 - x0) * grain_s).sum(axis=0).sum() + np.abs((x2 - x0) * grain_s).sum(axis=0).sum()
    cv = []
    x = (x0 + x1 + x2).sum(axis=1)
    for x_i in x:
        cv.append(x_i - 1)
        cv.append(1 - x_i)
    return f, cv

ub = [1] * dim
lb = [0] * dim

problem1 = ea.Problem(
        name='problem',
        M=1,
        maxormins=[1],
        Dim=dim,
        varTypes=[1] * dim,
        lb=lb,
        ub=ub,
        evalVars=func1)

field1 = ea.crtfld('BG', problem1.varTypes, problem1.ranges)
field2 = ea.crtfld('BG', problem1.varTypes, problem1.ranges)
field3 = ea.crtfld('BG', problem1.varTypes, problem1.ranges)
population = [
    Population(Encoding='BG', Field=field1, NIND=50),
    Population(Encoding='BG', Field=field2, NIND=50),
    Population(Encoding='BG', Field=field3, NIND=50)
]

algorithm1 = ea.soea_multi_SEGA_templet(
    problem1,
    population,
    MAXGEN=1000,
    logTras=1,
    trappedValue=1e-7,
    maxTrappedCount=10)

res = ea.optimize(algorithm1,
                  seed=42,
                  verbose=False,
                  drawing=0,
                  outputMsg=True,
                  drawLog=False,
                  saveFlag=False)

res1 = res.get('Vars')[:, :130].reshape(26, 5)
res2 = res.get('Vars')[:, 130:].reshape(26, 5)

# 将粮食豆类结果写入excel
pd.DataFrame(x0, index=history_crop_to_land.iloc[:26, 0], columns=grain_bean).to_excel('./results/豆类/2023-豆类1.xlsx')
pd.DataFrame(res1, index=history_crop_to_land.iloc[:26, 0], columns=grain_bean).to_excel('./results/豆类/2024-豆类1.xlsx')
pd.DataFrame(res2, index=history_crop_to_land.iloc[:26, 0], columns=grain_bean).to_excel('./results/豆类/2025-豆类1.xlsx')


# 处理蔬菜豆类
vege_bean = ['豇豆', '刀豆', '芸豆']
vege_bean_dict = {key: value for key, value in zip(vege_bean, range(3))}
x0 = np.zeros((28, 3))
history_vege_to_land = history_crop_to_land.iloc[26:].groupby('地块名称')
k = -1
for group_name, group in history_vege_to_land:
    k += 1
    for row in group.iterrows():
        bean_name = row[1]['作物名称']
        if bean_name in vege_bean:
            x0[k, vege_bean_dict[bean_name]] = 1
            break
        else:
            continue

vege_s = land_to_area1['地块面积'].to_numpy()[26:].reshape(-1, 1)

dim2 = 28 * 3 * 2

@ea.Problem.single
def func2(x):
    x1 = x[:84].reshape(28, 3)
    x2 = x[84:].reshape(28, 3)
    f = np.abs((x1 - x0) * vege_s).sum(axis=0).sum() + np.abs((x2 - x0) * vege_s).sum(axis=0).sum()
    cv = []
    x = (x0 + x1 + x2).sum(axis=1)
    for x_i in x:
        cv.append(x_i - 1)
        cv.append(1 - x_i)
    return f, cv

ub = [1] * dim2
lb = [0] * dim2

problem2 = ea.Problem(
        name='problem',
        M=1,
        maxormins=[1],
        Dim=dim2,
        varTypes=[1] * dim2,
        lb=lb,
        ub=ub,
        evalVars=func2)

field1 = ea.crtfld('BG', problem2.varTypes, problem2.ranges)
field2 = ea.crtfld('BG', problem2.varTypes, problem2.ranges)
field3 = ea.crtfld('BG', problem2.varTypes, problem2.ranges)
population = [
    Population(Encoding='BG', Field=field1, NIND=50),
    Population(Encoding='BG', Field=field2, NIND=50),
    Population(Encoding='BG', Field=field3, NIND=50)
]

algorithm2 = ea.soea_multi_SEGA_templet(
    problem2,
    population,
    MAXGEN=1000,
    logTras=1,
    trappedValue=1e-7,
    maxTrappedCount=10)

res = ea.optimize(algorithm2,
                  seed=42,
                  verbose=False,
                  drawing=0,
                  outputMsg=True,
                  drawLog=False,
                  saveFlag=False)

res1 = res.get('Vars')[:, :84].reshape(28, 3)
res2 = res.get('Vars')[:, 84:].reshape(28, 3)

# 将蔬菜豆类结果写入excel
pd.DataFrame(x0, columns=vege_bean, index=land_to_area1['地块名称'][26:]).to_excel('./results/豆类/2023-豆类2.xlsx')
pd.DataFrame(res1, columns=vege_bean, index=land_to_area1['地块名称'][26:]).to_excel('./results/豆类/2024-豆类2.xlsx')
pd.DataFrame(res2, columns=vege_bean, index=land_to_area1['地块名称'][26:]).to_excel('./results/豆类/2025-豆类2.xlsx')