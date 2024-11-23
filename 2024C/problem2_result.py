"""
problem2_result.py 用于处理问题2的结果，包括作物利润比例分布和作物利润趋势。
主要功能包括：
- 生成某一年作物利润比例分布的饼图
- 生成某一作物利润趋势的折线图
"""

from matplotlib import pyplot as plt
from tools import *

plt.rc('font', family='SimHei')
plt.rc('axes', unicode_minus=False)


def Y(x, k, J):
    if X_t(x, k, J) <= R(k, crop_sale):
        return X_t(x, k, J) * P(k, crop_price)
    else:
        return R(k, crop_sale) * P(k, crop_price) + (X_t(x, k, J) - R(k, crop_sale)) * P(k, crop_price) * 0.5

def X_t(x, k, J):
    sum_x = 0
    for j in J:
        sum_x += x[j - J[0]] * S(j) * Q(j, k, crop_land_to_yield)
    return sum_x

def one_year_crop_profit(crop, data):
    J = data.iloc[:, 1].to_numpy() - 1
    x = data[crop].to_numpy()
    k = crop_index[crop]
    y = Y(x, k, J)
    sum_jk = 0
    for j in J:
        sum_jk += x[j - J[0]] * C(j, k) * S(j)
    return y - sum_jk


def one_year_crops_profit(year):
    path = f'./results/2-{year}'
    crops_profit = {}
    all_data = [
        pd.read_excel(f'{path}/{year}-粮食.xlsx'),
        pd.read_excel(f'{path}/{year}-蔬菜.xlsx'),
        pd.read_excel(f'{path}/{year}-萝卜-第二季.xlsx'),
        pd.read_excel(f'{path}/{year}-蔬菜-第二季.xlsx'),
        pd.read_excel(f'{path}/{year}-豆类1.xlsx'),
        pd.read_excel(f'{path}/{year}-豆类2.xlsx'),
        pd.read_excel(f'{path}/{year}-食用菌-第二季.xlsx'),
    ]
    for data in all_data:
        for crop in data.columns[2:]:
            if crop not in crops_profit.keys():
                crops_profit[crop] = one_year_crop_profit(crop, data)
            else:
                crops_profit[crop] += one_year_crop_profit(crop, data)
    return crops_profit


def one_year_crop_profit_pie(title, save_path, year):
    crops_profit = one_year_crops_profit(year)
    crops_profit = pd.DataFrame({'crop': crops_profit.keys(), 'profit': crops_profit.values()})
    crops_profit[crops_profit['profit'] < 0] = 0
    crops_profit_sorted = crops_profit.sort_values('profit', ascending=False)

    top_10 = crops_profit_sorted.head(10)
    others = pd.DataFrame({'crop': ['其他'], 'profit': [crops_profit_sorted.iloc[10:]['profit'].sum()]})
    df_pie = pd.concat([top_10, others])
    colors = plt.cm.Spectral(np.linspace(0.1, 0.9, len(df_pie)))
    fig, ax = plt.subplots(figsize=(12, 8))
    wedges, texts, autotexts = ax.pie(df_pie['profit'],
                                      labels=df_pie['crop'],
                                      colors=colors,
                                      autopct='%1.1f%%',
                                      pctdistance=0.85,
                                      labeldistance=1.05)
    plt.setp(texts, size=10)
    plt.setp(autotexts, size=9, weight="bold", color="white")
    ax.set_title(title, fontsize=16)
    ax.legend(wedges, df_pie['crop'],
              title="作物",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1),
              fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def one_crop_profit_trend(crop, start, end):
    profit = []
    for year in range(start, end + 1):
        profit.append(one_year_crops_profit(year)[crop])
    plt.figure(figsize=(12, 6))
    plt.plot(range(start, end + 1), profit)
    plt.title(crop + '利润趋势', fontsize=16)
    plt.xlabel('年份', fontsize=12)
    plt.ylabel('利润', fontsize=12)
    plt.savefig('./pictures/' + crop + '利润趋势.png')
    plt.close()


one_year_crop_profit_pie('2024作物利润比例分布（前10种）', './pictures/3-2024作物利润比例分布.png', 2024)
one_year_crop_profit_pie('2025作物利润比例分布（前10种）', './pictures/3-2025作物利润比例分布.png', 2025)
one_year_crop_profit_pie('2026作物利润比例分布（前10种）', './pictures/3-2026作物利润比例分布.png', 2026)
one_year_crop_profit_pie('2027作物利润比例分布（前10种）', './pictures/3-2027作物利润比例分布.png', 2027)
one_year_crop_profit_pie('2028作物利润比例分布（前10种）', './pictures/3-2028作物利润比例分布.png', 2028)
one_year_crop_profit_pie('2029作物利润比例分布（前10种）', './pictures/3-2029作物利润比例分布.png', 2029)
one_year_crop_profit_pie('2030作物利润比例分布（前10种）', './pictures/3-2030作物利润比例分布.png', 2030)


# one_crop_profit_trend('小麦', 2024, 2030)
# one_crop_profit_trend('玉米', 2024, 2030)
# one_crop_profit_trend('谷子', 2024, 2030)
# one_crop_profit_trend('高粱', 2024, 2030)
# one_crop_profit_trend('黍子', 2024, 2030)
# one_crop_profit_trend('荞麦', 2024, 2030)
# one_crop_profit_trend('南瓜', 2024, 2030)
# one_crop_profit_trend('红薯', 2024, 2030)
# one_crop_profit_trend('莜麦', 2024, 2030)
# one_crop_profit_trend('大麦', 2024, 2030)
# one_crop_profit_trend('水稻', 2024, 2030)

profit_2023 = [crop_sale[key] * crop_price[key] for key in crop_sale.keys()]

