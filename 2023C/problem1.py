import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler

plt.rc('font', family='SimHei')
plt.rc('axes', unicode_minus=False)

# 读取文件
data1 = pd.DataFrame(pd.read_excel('data/附件1.xlsx'))
data2 = pd.DataFrame(pd.read_excel('data/附件2.xlsx'))

# 建立单品编号与单品类别的映射字典
data1_map = data1.set_index('单品编码')['单品名称'].to_dict()
data1_category = {}
category1 = []
category2 = []
category3 = []
category4 = []
category5 = []
category6 = []
for i in range(data1.shape[0]):
    if data1.iloc[i, 2] == 1011010101:
        category1.append(data1.iloc[i, 0])
    elif data1.iloc[i, 2] == 1011010201:
        category2.append(data1.iloc[i, 0])
    elif data1.iloc[i, 2] == 1011010402:
        category3.append(data1.iloc[i, 0])
    elif data1.iloc[i, 2] == 1011010501:
        category4.append(data1.iloc[i, 0])
    elif data1.iloc[i, 2] == 1011010504:
        category5.append(data1.iloc[i, 0])
    else:
        category6.append(data1.iloc[i, 0])

data1_category['花叶类'] = category1
data1_category['花菜类'] = category2
data1_category['水生根茎类'] = category3
data1_category['茄类'] = category4
data1_category['辣椒类'] = category5
data1_category['食用菌'] = category6

# 统计各个单品每日的销售量和定价
item_num = data1.iloc[:, 0]

# 记录时间戳
times = []
time = data2.iloc[0, 0]
single_day = {}
single_sale = {}

for i in range(len(item_num)):
    single_sale[item_num[i]] = 0
for i in range(len(item_num)):
    single_day[item_num[i]] = 0

data11 = pd.DataFrame(pd.read_excel('data/销售量统计.xlsx'))
data14 = data11.copy()
count = 1

for i in range(data2.shape[0]):
    temp_time = data2.iloc[i, 0]

    if time != temp_time:
        time = temp_time
        for j in range(data11.shape[0]):
            data11.iloc[j, count] = single_day[data11.iloc[j, 0]]
            data14.iloc[j, count] = single_sale[data11.iloc[j, 0]]
        for k in range(len(item_num)):
            single_day[item_num[k]] = 0
        count += 1

    single_day[data2.iloc[i, 2]] += data2.iloc[i, 3]
    single_sale[data2.iloc[i, 2]] = data2.iloc[i, 4]

    if i == data2.shape[0] - 1:
        for j in range(data11.shape[0]):
            data11.iloc[j, count] = single_day[data11.iloc[j, 0]]
            data14.iloc[j, count] = single_sale[data11.iloc[j, 0]]

    if temp_time not in times:
        times.append(temp_time)

data11.to_excel('result/problem1/单品单日销售量统计.xlsx')
data14.to_excel('result/problem2/单品单日定价统计.xlsx')

# 绘制每个单品的销售量随时间变化图
for i in range(251):
    plt.figure(num=i)
    plt.plot(times, data11.iloc[i, 1:], c='#4682B4')
    title = data1_map[data11.iloc[i, 0]]
    plt.title(title)
    plt.savefig('result/problem1/pictures1/' + title + '.png', dpi=400)
    plt.close()

# 统计每个品类每天的销售量
category = list(data1_category.keys())
categories = {}
for i in range(len(category)):
    categories[category[i]] = 0

data12 = pd.DataFrame(pd.read_excel('data/品类统计.xlsx'))

for j in range(1, data11.shape[1]):
    for i in range(data11.shape[0]):
        cate = None
        for key in data1_category.keys():
            if data11.iloc[i, 0] in data1_category[key]:
                cate = key
        categories[cate] += data11.iloc[i, j]
    for k in range(data12.shape[0]):
        data12.iloc[k, j] = categories[data12.iloc[k, 0]]
    for k in range(len(category)):
        categories[category[k]] = 0

data12.to_excel('result/problem1/单日品类销售统计.xlsx')

# 绘制每个品类的销售量随时间变化图
for i in range(6):
    plt.figure(num=(i + 252))
    plt.plot(times, data12.iloc[i, 1:], c='#4682B4')
    title = data12.iloc[i, 0]
    plt.title(title)
    plt.savefig('result/problem1/pictures2/' + str(title) + '.png', dpi=400)
    plt.close()

# 计算品类销售量列和
col_sum = []
for j in range(1, data12.shape[1]):
    mini_sum = 0
    for i in range(data12.shape[0]):
        mini_sum += data12.iloc[i, j]
    col_sum.append(mini_sum)

# 绘制单品堆积柱状图
colors = ['blue', 'green', 'black', 'red', 'yellow', 'purple']
plt.figure(num=252, figsize=(10, 8))

percentages = []
for i in range(data12.shape[0]):
    percentage = data12.iloc[i, 1:] / col_sum
    percentages.append(percentage)
for i in range(data12.shape[0]):
    if i == 0:
        plt.bar(times, percentages[0], color=colors[0], width=0.4, label=data12.iloc[i, 0], zorder=5)
    else:
        plt.bar(times, percentages[i], bottom=percentages[i - 1], color=colors[i], width=0.4, label=data12.iloc[i, 0],
                zorder=5)
plt.ylim(0, 1.01)
plt.yticks(np.arange(0, 1.2, 0.2), [f'{i}' for i in range(0, 120, 20)])
plt.grid(axis='y', alpha=0.5, ls='--')
plt.legend(frameon=False, bbox_to_anchor=(1.01, 1))
plt.tight_layout()
plt.legend()
plt.savefig('result/problem1/pictures3/品类堆积柱状图.png', dpi=600)

# 计算单品和品类的相关系数矩阵
scaler = MinMaxScaler()
corr1 = np.corrcoef(scaler.fit_transform(data11.iloc[:, 1:]))
corr2 = np.corrcoef(scaler.fit_transform(data12.iloc[:, 1:]))

# 绘制热力图
plt.figure(num=500)
fig1 = sns.heatmap(corr1, annot=False, vmin=-1, vmax=1, square=True, cmap="Blues")
fig1.get_figure().savefig('result/problem1/pictures4/单品相关系数热力图.png', bbox_inches='tight', transparent=True,
                          dpi=600)

plt.figure(num=501)
fig2 = sns.heatmap(corr2, annot=False, vmin=-1, vmax=1, square=True, cmap="Blues")
fig2.get_figure().savefig('result/problem1/pictures4/品类相关系数热力图.png', bbox_inches='tight', transparent=True,
                          dpi=600)

# 进行品类的ADF检验
data13 = pd.DataFrame(columns=['ADF Statistic', 'p-value'])
for i in range(0, data12.shape[0]):
    adf_result = adfuller(data12.iloc[i, 1:])
    data_13 = {'ADF Statistic': adf_result[0], 'p-value': adf_result[1]}
    data13 = pd.concat([data13, pd.DataFrame([data_13])], ignore_index=True)
data13.to_excel('result/problem1/品类ADF检验结果.xlsx')
