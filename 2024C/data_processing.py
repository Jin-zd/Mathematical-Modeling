"""
data_processing.py 用于处理农作物种植和销售数据，并生成相关的图表和分析结果。

主要功能包括：
- 读取和处理原始数据文件，生成新的数据文件。
- 计算农作物的销售额和利润，并生成相关的图表。
- 分析和可视化农作物种植面积和地块面积。
- 生成各类农作物和地块的比例分布图。
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# 读取数据
land_data = pd.read_excel('./data/附件1.xlsx', sheet_name=0)
cropper_info = pd.read_excel('./data/附件1.xlsx', sheet_name=1)
copper_plant_data = pd.read_excel('./data/附件2.xlsx', sheet_name=0)
total_data = pd.read_excel('./data/附件2.xlsx', sheet_name=1)

# 生成新数据
land_data.rename(columns={'地块面积/亩': '地块面积'}, inplace=True)
pd.DataFrame(land_data.iloc[:, 0: 3]).to_excel('./new_data/data1.xlsx', index=False)

cropper_info_filled = cropper_info.ffill()
cropper_info_filled.rename(columns={'种植耕地': '种植耕地和季次'}, inplace=True)
cropper_info_filled.iloc[:, 1: 4].to_excel('./new_data/data2.xlsx', index=False)

copper_plant_data.rename(columns={'种植地块':'地块名称', '种植面积/亩': '作物种植面积', '种植季次':'作物季次'}, inplace=True)
copper_plant_data = copper_plant_data.ffill()
del copper_plant_data['作物编号']
copper_plant_data.to_excel('./new_data/data3.xlsx', index=False)

total_data.rename(columns={'亩产量/斤': '亩产量','种植成本/(元/亩)':'种植成本', '销售单价/(元/斤)': '销售单价'}, inplace=True)
for i in range(len(total_data)):
    price = total_data.loc[i, '销售单价']
    start, end = price.split('-')
    total_data.loc[i, '销售单价'] = (float(start) + float(end)) / 2
total_data.iloc[:, 2:].to_excel('./new_data/data4.xlsx', index=False)


prices = total_data['销售单价'].dropna().to_list()
price_matrix = []

for price in prices:
    start, end = price.split('-')
    start = float(start)
    end = float(end)
    mid = (start + end) / 2
    price_matrix.append([start, mid, end])

price_matrix = np.array(price_matrix)
product_num = total_data['亩产量'].dropna().to_numpy().reshape(-1, 1)

result = price_matrix * product_num

total_data = total_data.assign(
    最低销售额=result[:, 0],
    平均销售额=result[:, 1],
    最高销售额=result[:, 2]
)
total_data['平均销售额'] = total_data['平均销售额'] - total_data['亩产量']
sorted_df = total_data.groupby('地块类型', group_keys=False).apply(lambda x: x.sort_values('平均销售额'))

crops = total_data['作物名称']
profits = total_data['平均销售额']


df = pd.DataFrame({'crop': crops, 'profit': profits})
df_sorted = df.sort_values('profit', ascending=False)
plt.rc('font', family='SimHei')
plt.rc('axes', unicode_minus=False)

# 2023农作物平均每亩利润
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(df_sorted['crop'], df_sorted['profit'])
ax.set_title('2023农作物平均每亩利润', fontsize=16)
ax.set_xlabel('农作物类型', fontsize=12)
ax.set_ylabel('平均每亩利润（元）', fontsize=12)
plt.xticks(rotation=60, ha='right')
plt.tight_layout()
plt.savefig('./pictures/2023农作物平均每亩利润.png')
plt.close(fig)
sorted_df.to_excel('./data/附件2_处理.xlsx', index=False)


# 2023作物种植面积
crop_dict = copper_plant_data.groupby('作物名称')['作物种植面积'].sum().to_dict()
df = pd.DataFrame({'crop': crop_dict.keys(), 'area': crop_dict.values()})
df_sorted = df.sort_values('area', ascending=False)
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(df_sorted['crop'], df_sorted['area'])
ax.set_title('2023作物种植面积', fontsize=16)
ax.set_xlabel('作物名称', fontsize=12)
ax.set_ylabel('种植面积（亩）', fontsize=12)
plt.xticks(rotation=60, ha='right')
plt.tight_layout()
plt.savefig('./pictures/2023作物种植面积.png')
plt.close(fig)


# 作物种植面积分析
df_sorted['cumulative_percentage'] = df_sorted['area'].cumsum() / df_sorted['area'].sum()
fig, ax1 = plt.subplots(figsize=(12, 6))
bars = ax1.bar(df_sorted['crop'], df_sorted['area'])
ax1.set_xlabel('作物名称', fontsize=12)
ax1.set_ylabel('种植面积（亩）', fontsize=12)
plt.xticks(rotation=60, ha='right')
ax2 = ax1.twinx()
line = ax2.plot(df_sorted['crop'], df_sorted['cumulative_percentage'], 'b--', linewidth=2, marker='o')
ax2.set_ylabel('累积百分比 (%)', fontsize=12)
ax2.set_ylim(0, 1.1)
plt.title('2023作物种植面积分析', fontsize=16)
ax1.legend([bars], ['种植面积'], loc='lower right', bbox_to_anchor=(0.95, 0.5))
ax2.legend([line[0]], ['累积百分比'], loc='lower right', bbox_to_anchor=(0.95, 0.45))
plt.subplots_adjust(right=0.85)
plt.tight_layout()
plt.savefig('./pictures/2023作物种植面积分析.png')
plt.close(fig)

# 地块面积分析
df = pd.DataFrame({'land': land_data['地块名称'], 'area': land_data['地块面积']})
df_sorted = df.sort_values('area', ascending=False)
df_sorted['cumulative_percentage'] = df_sorted['area'].cumsum() / df_sorted['area'].sum()
fig, ax1 = plt.subplots(figsize=(12, 6))
bars = ax1.bar(df_sorted['land'], df_sorted['area'])
ax1.set_xlabel('地块名称', fontsize=12)
ax1.set_ylabel('地块面积（亩）', fontsize=12)
ax1.axhline(y=30, color='r', linestyle='--', alpha=0.5)
ax1.axhline(y=15, color='b', linestyle='--', alpha=0.5)
plt.xticks(rotation=60, ha='right')
ax2 = ax1.twinx()
line = ax2.plot(df_sorted['land'], df_sorted['cumulative_percentage'], 'b--', linewidth=2, marker='o')
ax2.set_ylabel('累积百分比 (%)', fontsize=12)
ax2.set_ylim(0, 1.1)
plt.title('地块面积分析', fontsize=16)
ax1.legend([bars], ['地块面积'], loc='lower right', bbox_to_anchor=(0.95, 0.5))
ax2.legend([line[0]], ['累积百分比'], loc='lower right', bbox_to_anchor=(0.95, 0.45))
plt.subplots_adjust(right=0.85)
plt.tight_layout()
plt.savefig('./pictures/地块面积分析.png')
plt.close(fig)


# 基本地块面积分析
df = pd.read_excel('./new_data/data5.xlsx')
df_sorted = pd.DataFrame({'land': df['小块'], 'area': df['地块面积/亩']}).sort_values('area', ascending=False)
df_sorted = df_sorted[df_sorted['area'] > 0.3]
plt.figure(figsize=(12, 6))
x = [str(i) for i in df_sorted['land'].to_list()]
plt.bar(x, df_sorted['area'].to_numpy())
plt.title('基本地块面积')
plt.xlabel('基本地块编号')
plt.ylabel('地块面积（亩）')
plt.tight_layout()
plt.savefig('./pictures/基本地块面积.png')
plt.close()


# 作物种植面积比例分布
top_10 = df_sorted.head(10)
others = pd.DataFrame({'crop': ['其他'], 'area': [df_sorted.iloc[10:]['area'].sum()]})
df_pie = pd.concat([top_10, others])
colors = plt.cm.Spectral(np.linspace(0.1, 0.9, len(df_pie)))
fig, ax = plt.subplots(figsize=(12, 8))
wedges, texts, autotexts = ax.pie(df_pie['area'],
                                  labels=df_pie['crop'],
                                  colors=colors,
                                  autopct='%1.1f%%',
                                  pctdistance=0.85,
                                  labeldistance=1.05)
plt.setp(texts, size=10)
plt.setp(autotexts, size=9, weight="bold", color="white")
ax.set_title("作物种植面积比例分布（前10种）", fontsize=16)
ax.legend(wedges, df_pie['crop'],
          title="作物",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1),
          fontsize=10)
plt.tight_layout()
plt.savefig('./pictures/2023作物种植面积比例分布.png')
plt.close(fig)


# 现有耕地资源分析
def autopct_format(values):
    def my_format(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        # 返回空字符串，如果是最小的两个值
        if val in sorted(values)[:2]:
            return ''
        return f'{pct:.1f}%'
    return my_format

land_dict = land_data.groupby('地块类型')['地块面积'].sum().to_dict()
df = pd.DataFrame({'type': land_dict.keys(), 'area': land_dict.values()})
df_sorted = df.sort_values('area', ascending=False)
colors = plt.cm.Spectral(np.linspace(0.1, 0.6, len(df_sorted)))

fig, ax = plt.subplots(figsize=(12, 8))
wedges, texts, autotexts = ax.pie(df_sorted['area'],
                                  labels=None,  # 移除饼图周围的标签
                                  colors=colors,
                                  autopct=autopct_format(df_sorted['area']),
                                  pctdistance=0.75)

plt.setp(autotexts, size=9, weight="bold", color="white")
ax.set_title("各类型耕地面积比例", fontsize=16)
ax.legend(wedges, df_sorted['type'],
          title="耕地类型",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1),
          fontsize=10)

plt.tight_layout()
plt.savefig('./pictures/各类型耕地面积比例.png')
plt.close(fig)

for i in range(len(cropper_info_filled)):
    text_info = cropper_info_filled.loc[i, '种植耕地和季次']
    text_split = [text for text in text_info.split('\n') if text]


