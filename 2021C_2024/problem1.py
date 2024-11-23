import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pyrepo_mcda.additions import rank_preferences
from pyrepo_mcda.mcda_methods import VIKOR
from pyrepo_mcda import normalizations as norms
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.model_selection import train_test_split

# 读取数据
weekly_order = pd.read_excel('./data/附件1 近5年402家供应商的相关数据.xlsx', sheet_name=0)
weekly_supply = pd.read_excel('./data/附件1 近5年402家供应商的相关数据.xlsx', sheet_name=1)

# 去除供货周数小于等于 5 的供应商
week_sum = weekly_supply.iloc[:, 2:].apply(lambda x: (x != 0).sum(), axis=1)
weekly_order = weekly_order[week_sum > 5]
weekly_supply = weekly_supply[week_sum > 5]

# 供货总量
type_rate_dict = {'A': 0.6, 'B': 0.66, 'C': 0.72}
supply_sum = weekly_supply.iloc[:, 2:].sum(axis=1).to_list()
for i in range(len(supply_sum)):
    supply_sum[i] = supply_sum[i] / type_rate_dict[weekly_supply.iloc[i, 1]]
pd.DataFrame({'供应商ID': weekly_supply.iloc[:, 0], '供货总量': supply_sum}).to_excel('./results/supply_sum.xlsx', index=False)

# 供货稳定性（基于变异系数）
cv = weekly_supply.iloc[:, 2:].apply(lambda x: x.std() / x.mean(), axis=1)

# 交货率和交货稳定性
supply_rate = []
tranche = []
for i in range(weekly_order.shape[0]):
    order = weekly_order.iloc[i, 2:]
    supply = weekly_supply.iloc[i, 2:]
    no_zero_order = (order != 0).sum()
    rate = np.where(order != 0, supply / order, np.nan)
    rate = rate.astype(float)
    rate_without_nan = rate[~np.isnan(rate)]
    tranche.append((rate_without_nan.std() / rate_without_nan.mean()))
    supply_rate.append(np.nansum(rate) / no_zero_order)

# 供货周数
supply_week_num = [(weekly_supply.iloc[i, 2:] != 0).sum() for i in range(weekly_supply.shape[0])]

# 特征矩阵
values = pd.DataFrame({'供货总量': supply_sum, '供货稳定性': cv, '交货率': supply_rate, '交货稳定性': tranche, '供货周数': supply_week_num})

# K 均值聚类
k_data = values.values
k_data = (k_data - np.mean(k_data, axis=0)) / np.std(k_data, axis=0)
data_train, data_test = train_test_split(k_data, random_state=42)
kmeans = KMeans(n_clusters=4, random_state=21, n_init=10, max_iter=100)
kmeans.fit(data_train)
labels_test = kmeans.predict(data_test)
s = silhouette_score(data_test, labels_test)
chi = calinski_harabasz_score(data_test, labels_test)
labels = kmeans.predict(k_data)

print(f'轮廓系数:{s}, Calinski-Harabasz:{chi}')
kmeans_result = pd.DataFrame({'供应商ID': weekly_supply.iloc[:, 0], '类别': labels})
kmeans_result.to_excel('./results/kmeans_result.xlsx', index=False)

# 各个类别数量百分比的饼状图
labels_count = pd.Series(labels).value_counts()
labels_count = labels_count / labels_count.sum()
plt.rc('font', family='SimHei')
plt.rc('axes', unicode_minus=False)
plt.figure(figsize=(10, 6))
labels_count.plot.pie(autopct='%1.1f%%', colors=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99'],
              wedgeprops={'edgecolor': 'black', 'linewidth': 2 })
plt.title('各类别数量百分比')
plt.savefig('./pictures/labels_count_pie.png', dpi=600)
plt.close()

# 特征矩阵筛选
label = [2, 3]
k_values = pd.DataFrame({'供应商ID': weekly_supply.iloc[:, 0], '类别': labels, '供货总量': supply_sum, '供货稳定性': cv,
                         '交货率': supply_rate, '交货稳定性': tranche, '供货周数': supply_week_num})
final_k_values = k_values[k_values['类别'].isin(label)].reset_index(drop=True)
final_values = final_k_values.iloc[:, 2:].to_numpy()

# Critic 权重
normalized_matrix = (final_values - final_values.min(axis=0)) / (final_values.max(axis=0) - final_values.min(axis=0))
variances = np.var(normalized_matrix, axis=0)
diff_matrix = np.abs(normalized_matrix[:, np.newaxis] - normalized_matrix[np.newaxis, :])
avg_diff = np.mean(diff_matrix, axis=(0, 1))
weights = variances / avg_diff
weights = weights / np.sum(weights)

# VIKOR 多属性评价
criteria = np.array([1, -1, 1, -1, 1])
vikor_model = VIKOR(normalization_method=norms.minmax_normalization)
pref = vikor_model(final_values, weights, criteria)
rank = rank_preferences(pref, reverse=False)
rank_result = pd.DataFrame({'供应商ID': final_k_values.iloc[:, 0], '得分':pref, '排名': rank})
rank_result.to_excel('./results/rank_result.xlsx', index=False)

top50 = rank_result[rank_result['排名'] <= 50]

final_k_values.to_excel('./results/final_k_values.xlsx', index=False)
weekly_supply[weekly_supply['供应商ID'].isin(top50['供应商ID'])].reset_index(drop=True).to_excel('./results/final_supply.xlsx', index=False)
weekly_order[weekly_order['供应商ID'].isin(top50['供应商ID'])].reset_index(drop=True).to_excel('./results/final_order.xlsx', index=False)

weekly_supply = pd.DataFrame(pd.read_excel('./results/final_supply.xlsx'))

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

ax1.bar(x, y1, color='#71c9ce', alpha=0.7, label='supply sum')
ax1.tick_params(axis='y')

ax2 = ax1.twinx()
ax2.plot(x, y2, color='#aa96da', linestyle='--', label='supply rate')
ax2.set_ylim(0, 1.1)
ax2.tick_params(axis='y')

fig.legend()

plt.show()