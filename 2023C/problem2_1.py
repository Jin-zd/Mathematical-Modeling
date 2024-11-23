import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm

# 提取数据
data1 = pd.DataFrame(pd.read_excel('data/附件1.xlsx'))
data3 = pd.DataFrame(pd.read_excel('data/附件3.xlsx'))
data4 = pd.DataFrame(pd.read_excel('data/附件4.xlsx'))

data21 = pd.DataFrame(pd.read_excel('result/problem1/单品单日销售量统计.xlsx'))
data22 = pd.DataFrame(pd.read_excel('result/problem1/单日品类销售统计.xlsx'))
data23 = pd.DataFrame(pd.read_excel('result/problem2/单品单日定价统计.xlsx'))

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


# 返回给定编号所属的品类
def cate_select(number):
    for key in data1_category.keys():
        if number in data1_category[key]:
            return key


# 计算单品销量在对应品类中的比重
item_num = data1.iloc[:, 0]
cate_list = list(data1_category.keys())

row_sum = np.sum(data21.iloc[:, 2:], axis=1)
categories_sum = {}
for i in range(len(cate_list)):
    categories_sum[cate_list[i]] = []

for i in range(data21.shape[0]):
    cate = cate_select(data21.iloc[i, 1])
    categories_sum[cate].append(row_sum[i])

cate_sum = {}
for i in range(len(cate_list)):
    cate_sum[cate_list[i]] = np.sum(categories_sum[cate_list[i]])

rate = {}
for i in range(data21.shape[0]):
    cate = cate_select(data21.iloc[i, 1])
    rate[data21.iloc[i, 1]] = np.sum(data21.iloc[i, 2:]) / cate_sum[cate]

# 计算品类加权损耗率
loss_categories = {}
for i in range(len(cate_list)):
    loss_categories[cate_list[i]] = 0
for i in range(data4.shape[0]):
    num = data4.iloc[i, 0]
    cate_flag = cate_select(num)
    loss_categories[cate_flag] += data4.iloc[i, 2] * rate[num]

pd.DataFrame(loss_categories, index=[0]).T.to_excel('result/problem2/品类加权损耗率统计.xlsx')


# 计算加权和
def weight_sum(data, flag=True, n=0):
    return_data = pd.DataFrame(pd.read_excel('data/品类统计.xlsx'))

    temp_categories = {}
    for i in range(len(cate_list)):
        temp_categories[cate_list[i]] = 0

    if flag:
        n = 1
    for j in range(n + 1, data.shape[1]):
        for i in range(data.shape[0]):
            num = data.iloc[i, n]
            cate_flag = cate_select(num)
            temp_categories[cate_flag] += data.iloc[i, j] * rate[num]
        for k in range(return_data.shape[0]):
            return_data.iloc[k, j - n] = temp_categories[return_data.iloc[k, 0]]
        for k in range(len(cate_list)):
            temp_categories[cate_list[k]] = 0
    return return_data


# 计算品类单日加权定价
data25 = weight_sum(data23)
data25.to_excel('result/problem2/品类单日加权定价统计.xlsx')

# 统计单品每日进价
times = []
time = data3.iloc[0, 0]
count = 1
single_cost = {}
for i in range(len(item_num)):
    single_cost[item_num[i]] = 0
data26 = pd.DataFrame(pd.read_excel('data/销售量统计.xlsx'))
for i in range(data3.shape[0]):
    temp_time = data3.iloc[i, 0]

    if time != temp_time:
        time = temp_time
        for j in range(data26.shape[0]):
            if count < data26.shape[1]:
                data26.iloc[j, count] = single_cost[data26.iloc[j, 0]]
        for k in range(len(item_num)):
            single_cost[item_num[k]] = 0
        count += 1

    single_cost[data3.iloc[i, 1]] = data3.iloc[i, 2]

    if temp_time not in times:
        times.append(temp_time)

data26.to_excel('result/problem2/单品每日进价统计.xlsx')

# 计算品类每日加权定价
data26 = weight_sum(data26, flag=False)
data26.to_excel('result/problem2/品类加权进价统计.xlsx')

# 计算品类加权损失率
cate_loss = {}
for i in range(len(cate_list)):
    cate_loss[cate_list[i]] = 0

for i in range(data4.shape[0]):
    cate = cate_select(data4.iloc[i, 0])
    cate_loss[cate] += data4.iloc[i, 2] * rate[data4.iloc[i, 0]]

data27 = (data25.iloc[:, 1:] - data26.iloc[:, 1:]) / data26.iloc[:, 1:] / 100
data27.columns = range(1, 1086)
data27.to_excel('result/problem2/品类单日利润率统计.xlsx')

profit_rate = {'花叶类': data27.iloc[0, :], '花菜类': data27.iloc[1, :], '水生根茎类': data27.iloc[2, :],
               '茄类': data27.iloc[3, :], '辣椒类': data27.iloc[4, :], '食用菌': data27.iloc[5, :]}

# 进行Loess平滑处理
smoothed_sale = {}
for i in range(len(cate_list)):
    smoothed_sale[cate_list[i]] = []

for i in range(data22.shape[0]):
    loess_smoothed = sm.nonparametric.lowess(data22.iloc[i, 2:], range(1, 1086), frac=0.1)
    smoothed_x, smoothed_y = loess_smoothed.T

    title = data22.iloc[i, 1]
    smoothed_sale[title] = smoothed_y

    plt.figure(num=(600 + i))
    plt.rc('font', family='SimHei')
    plt.rc('axes', unicode_minus=False)

    plt.plot(range(1, 1086), data22.iloc[i, 2:], label='原始数据', alpha=0.5)
    plt.plot(smoothed_x, smoothed_y, color='red', label='Loess平滑')
    plt.title(title)
    plt.legend()
    plt.savefig('result/problem2/Loess平滑/' + str(title) + '.png', dpi=400)
    plt.close()

# 计算弹性销售量
flexible_sale = {}
for i in range(len(cate_list)):
    k = 0
    for j in range(data22.shape[0]):
        if data22.iloc[j, 1] == cate_list[i]:
            k = j
            break
    flexible_sale[cate_list[i]] = data22.iloc[k, 2:].to_numpy() - np.array(smoothed_sale[cate_list[i]])

# 绘制弹性销售量与利润率散点图
for i in range(len(cate_list)):
    title = cate_list[i]
    Y = flexible_sale[title]
    X = profit_rate[title]

    Y = list(Y)
    X = list(X)

    plt.figure(num=(700 + i))
    plt.rc('font', family='SimHei')
    plt.rc('axes', unicode_minus=False)
    plt.scatter(X, Y, marker='o', s=1, color='red', label='数据点')
    plt.xlabel('利润率')
    plt.ylabel('弹性销售量')
    plt.xlim(-1, 1)
    plt.legend()
    plt.title(title)
    plt.savefig('result/problem2/弹性销售量与利润率散点图/' + str(title) + '.png', dpi=400)

data27 = pd.DataFrame(pd.read_excel('result/problem2/品类单日加权定价统计.xlsx'))

# 对时间序列进行差分操作
rate_diff = {}
sale_diff = {}
for i in range(len(cate_list)):
    k = 0
    for j in range(data22.shape[0]):
        if data22.iloc[j, 1] == cate_list[i]:
            k = j
            break
    a = data22.iloc[i, 2:]
    b = data27.iloc[i, 2:]
    a = a.fillna(a.mean()).to_numpy()
    b = b.fillna(a.mean()).to_numpy()
    diff_sale = np.diff(a)
    diff_rate = np.diff(b)
    den1 = data22.iloc[i, :].to_numpy()[2:data22.shape[1] - 1]
    den2 = data27.iloc[i, :].to_numpy()[2:data27.shape[1] - 1]

    sale_diff[cate_list[i]] = diff_sale / den1
    rate_diff[cate_list[i]] = diff_rate / den2


# 绘制销售量比率与定价比率散点图
for i in range(len(cate_list)):
    title = cate_list[i]
    Y = sale_diff[title]
    X = rate_diff[title]

    Y = list(Y)
    X = list(X)

    for j in range(len(X)):
        if math.isnan(X[j]) or math.isinf(X[j]):
            X[j] = np.median(X)
        if math.isnan(Y[j]) or math.isinf(Y[j]):
            Y[j] = np.median(Y)
    df = {'y': np.array(Y), 'x': np.array(X)}
    model1 = sm.formula.ols('y~x', data=df).fit()
    params = model1.params

    x = np.linspace(-1, 1, 1000)
    y1 = params[0] + params[1] * x

    plt.figure(num=(800 + i))
    plt.rc('font', family='SimHei')
    plt.rc('axes', unicode_minus=False)
    plt.scatter(X, Y, marker='o', s=1, color='red', label='数据点')
    plt.plot(x, y1, color='orange', label='OLS拟合')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.legend()
    plt.title(title)
    plt.savefig('result/problem2/销售量与定价散点图/' + str(title) + '.png', dpi=400)
