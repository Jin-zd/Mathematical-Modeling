import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

weekly_order = pd.read_excel('./data/附件1 近5年402家供应商的相关数据.xlsx', sheet_name=0)
weekly_supply = pd.read_excel('./data/附件1 近5年402家供应商的相关数据.xlsx', sheet_name=1)

weekly_order_groups = weekly_order.groupby('材料分类')
type_sum = [group.iloc[:, 2:].sum(axis=0) for name, group in weekly_order_groups]

type_sum_df = pd.DataFrame(type_sum).transpose()
type_sum_df.columns = [name for name, group in weekly_order_groups]
type_sum_df.to_excel('./results/type_sum.xlsx', index=False)

def plot_line(x, y, label=None, xlabel=None, ylabel=None, title=None, save_path=None):
    plt.rc('font', family='SimHei')
    plt.rc('axes', unicode_minus=False)
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, label=label, c='#A07EE7')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(save_path, dpi=600)
    plt.close()



x = range(weekly_supply.shape[0])
total = weekly_supply.iloc[:, 2:].sum(axis=1).to_numpy()
y1 = np.sort(total)[::-1]
rate = y1 / y1.sum()
y2 = []
for i in range(len(rate)):
    if i == 0:
        y2.append(rate[i])
    else:
        y2.append(rate[i] + y2[i - 1])

fig, ax1 = plt.subplots()

ax1.bar(x, y1, color='#00adb5', alpha=0.7, label='supply sum')
ax1.tick_params(axis='y')

ax2 = ax1.twinx()
ax2.plot(x, y2, color='blue', linestyle='--', label='supply rate')
ax2.set_ylim(0, 1)
ax2.tick_params(axis='y')

fig.legend()

plt.show()

# # 绘制每周各个类别的总和折线图
# week_num = weekly_supply.shape[1] - 2
# for i, column in enumerate(type_sum_df.columns):
#     plot_line(range(week_num), type_sum_df[column], column, '周数', '总和',
#                  f'材料分类 {column} 总和折线图', './pictures/type_order_sum/type_' + str(column) + '.png')
#
# # 绘制每个供应商的周供货折线图
# for i in range(weekly_supply.shape[0]):
#     plot_line(range(week_num), weekly_supply.iloc[i, 2:], weekly_supply.iloc[i, 1], '周数', '供应量',
#                  f'供应商 {weekly_supply.iloc[i, 0]} 周供货折线图', './pictures/supply/' + weekly_supply.iloc[i, 0] + '.png')

