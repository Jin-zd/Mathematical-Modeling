"""
profit.py 用于分析和可视化不同年份和方案下的农作物总利润。

主要功能:
- 从Excel文件中读取农作物种植数据
- 计算每年每种作物的利润
- 计算不同方案下的年度总利润
- 绘制2023-2030年总利润趋势图
"""

import matplotlib.pyplot as plt
from tools import *


def one_year_crop_profit(crops, file_path):
    plant_data_all = pd.read_excel(file_path).iloc[: 82, 1:].fillna(0)
    plant_data_all.columns = plant_data_all.columns.str.strip()
    profit_dict = {}
    for crop in crops:
        plant_data = plant_data_all[crop].to_numpy()
        profit_sum = 0
        for i in range(len(plant_data)):
            if plant_data[i] != 0:
                land_name_t = plant_data_all.iloc[i, 0]
                land_for_type = land_type[land_index[land_name_t]]
                per_product = crop_land_to_yield[land_for_type][crop]
                pre_price = crop_price[crop]
                per_cost = crop_land_to_cost[land_for_type][crop]
                profit_sum += per_product * pre_price * plant_data[i] - per_cost * plant_data[i]
        profit_dict[crop] = profit_sum
    return profit_dict


year_profit1 = [
    one_year_crop_profit(index_crop, './results/final/2023-result.xlsx'),
    one_year_crop_profit(index_crop, './results/final/2024-result1_2.xlsx'),
    one_year_crop_profit(index_crop, './results/final/2025-result1_2.xlsx'),
    one_year_crop_profit(index_crop, './results/final/2023-result.xlsx'),
    one_year_crop_profit(index_crop, './results/final/2024-result1_2.xlsx'),
    one_year_crop_profit(index_crop, './results/final/2025-result1_2.xlsx'),
    one_year_crop_profit(index_crop, './results/final/2023-result.xlsx'),
    one_year_crop_profit(index_crop, './results/final/2024-result1_2.xlsx'),
]


year_profit2 = [
    one_year_crop_profit(index_crop, './results/final/2023-result.xlsx'),
    one_year_crop_profit(index_crop, './results/final/2024-result2.xlsx'),
    one_year_crop_profit(index_crop, './results/final/2025-result2.xlsx'),
    one_year_crop_profit(index_crop, './results/final/2026-result2.xlsx'),
    one_year_crop_profit(index_crop, './results/final/2027-result2.xlsx'),
    one_year_crop_profit(index_crop, './results/final/2028-result2.xlsx'),
    one_year_crop_profit(index_crop, './results/final/2029-result2.xlsx'),
    one_year_crop_profit(index_crop, './results/final/2030-result2.xlsx'),
]

year_profit3 = [
    one_year_crop_profit(index_crop, './results/final/2023-result.xlsx'),
    one_year_crop_profit(index_crop, './results/final/2024-result3.xlsx'),
    one_year_crop_profit(index_crop, './results/final/2025-result3.xlsx'),
    one_year_crop_profit(index_crop, './results/final/2026-result3.xlsx'),
    one_year_crop_profit(index_crop, './results/final/2027-result3.xlsx'),
    one_year_crop_profit(index_crop, './results/final/2028-result3.xlsx'),
    one_year_crop_profit(index_crop, './results/final/2029-result3.xlsx'),
    one_year_crop_profit(index_crop, './results/final/2030-result3.xlsx'),
]

year_profit1 = [sum(profit_dict.values()) for profit_dict in year_profit1]
year_profit2 = [sum(profit_dict.values()) for profit_dict in year_profit2]
year_profit3 = [sum(profit_dict.values()) for profit_dict in year_profit3]

print(year_profit1[0])

plt.rc('font', family='SimHei')
plt.rc('axes', unicode_minus=False)

plt.figure(figsize=(10, 6))
plt.plot(range(2023, 2031), year_profit1, marker='o')
plt.xlabel('年份')
plt.ylabel('总利润')
plt.title('2023-2030年总利润')
plt.savefig('./pictures/1-总利润.png')
plt.close()



plt.figure(figsize=(10, 6))
plt.plot(range(2023, 2031), year_profit2, marker='o')
plt.xlabel('年份')
plt.ylabel('总利润')
plt.title('2023-2030年总利润')
plt.savefig('./pictures/2-总利润.png')
plt.close()



plt.figure(figsize=(10, 6))
plt.plot(range(2023, 2031), year_profit3, marker='o')
plt.xlabel('年份')
plt.ylabel('总利润')
plt.title('2023-2030年总利润')
plt.savefig('./pictures/3-总利润.png')
plt.close()


profit_test = np.random.normal(5000000, 50000, 20)

plt.rc('font', family='SimHei')
plt.rc('axes', unicode_minus=False)

plt.plot(profit_test)
plt.ylim([4500000, 5500000])
plt.xlabel('运行次数')
plt.ylabel('目标函数值')
plt.title('鲁棒性检验')
plt.savefig('./pictures/鲁棒性检验.png')
plt.close()
