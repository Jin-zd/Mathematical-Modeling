import numpy as np
import pandas as pd
import geatpy as ea
import matplotlib.pyplot as plt

# 读取数据
weekly_supply = pd.DataFrame(pd.read_excel('./results/final_supply.xlsx'))

loss_rate = pd.DataFrame(pd.read_excel('./data/附件2 近5年8家转运商的相关数据.xlsx'))
mean_loss_rate = loss_rate.iloc[:, 1:].sum().sum() / (loss_rate.iloc[:, 1:] != 0).astype(int).sum().sum() / 100

select = [1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1,
          0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
          1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1,
          1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]

supply_type_dict = weekly_supply.set_index('供应商ID')['材料分类'].to_dict()
type_rate_dict = {'A': 0.6, 'B': 0.66, 'C': 0.72}


weekly_supply['选择'] = select
weekly_supply = weekly_supply[weekly_supply['选择'] == 1]
weekly_supply = weekly_supply.drop(columns='选择')

supply_k = []
for i in range(len(weekly_supply)):
    supply_k.append(type_rate_dict[supply_type_dict[weekly_supply.iloc[i, 0]]])
supply_k = np.array(supply_k)

price_rate_dict = {'A': 1.2, 'B': 1.1, 'C': 1}
supply_m = []
for i in range(len(weekly_supply)):
    supply_m.append(price_rate_dict[supply_type_dict[weekly_supply.iloc[i, 0]]])
supply_m = np.array(supply_m).reshape(42, 1)

n = 42 * 24

v = pd.DataFrame(pd.read_excel('./results/pred_weekly_supply.xlsx'))
v['选择'] = select
v = v[v['选择'] == 1]
v = v.drop(columns='选择')
pred = v.copy()
pred.iloc[:, 1:] = v.iloc[:, 1:].astype(int)
pred.to_excel('./results/select.xlsx', index=False)
v = v.iloc[:, 1:].to_numpy().astype(int)
v[v <= 0] = 0

# 订购方案
@ea.Problem.single
def func(x):
    x = x.reshape(42, 24)
    cv = []
    r = []

    row_mean = np.nanmean(x, axis=1)
    nan_indices = np.where(np.isnan(x))
    x[nan_indices] = np.take(row_mean, nan_indices[0])

    f1 = (x * supply_m).sum()
    f2 = x.sum()
    f = np.hstack([f1, f2])

    for k in range(24):
        if k == 0:
            r.append(28200)
        else:
            r.append(r[k - 1] + (x[:, k - 1].T / supply_k).sum() * (1 - mean_loss_rate) - 28200)

    for j in range(24):
        cv.append(28200 - (x[:, j].T / supply_k).sum() * (1 - mean_loss_rate) - r[j])
        cv.append(x[:, j].T.sum() - 48000)
        cv.append(-r[j])
    return f, cv


ub = []
lb = [0] * n

for i in range(v.shape[0]):
    for j in range(v.shape[1]):
        ub.append(v[i, j])

problem = ea.Problem(name='problem',
                         M=2,
                         maxormins=[1, 1],
                         Dim=n,
                         varTypes=[0] * n,
                         lb=lb,
                         ub=ub,
                         evalVars=func)

algorithm = ea.moea_NSGA3_templet(
    problem,
    ea.Population(Encoding='RI', NIND=100),
    MAXGEN=10000,
    logTras=1)

res = ea.optimize(algorithm,
                  verbose=False,
                  drawing=1,
                  outputMsg=True,
                  drawLog=False,
                  saveFlag=False)

select_supply_id = weekly_supply['供应商ID'].reset_index(drop=True)


row_sum = []
for i in range(len(res.get('Vars'))):
    row_sum.append(np.sum(res.get('Vars')[i]))

res_dict = dict(zip(row_sum, res.get('Vars')))
best_x = res_dict[min(row_sum)].reshape(42, 24)

order_res = pd.DataFrame(best_x.astype(int), columns=[f'week{i + 1}' for i in range(24)])
order_res.insert(0, '供应商ID', select_supply_id)
order_res.to_excel('./results/3-订购/weekly_order.xlsx', index=False)


problem2_order = pd.DataFrame(pd.read_excel('./results/2-订购/weekly_order.xlsx'))
problem3_order = pd.DataFrame(pd.read_excel('./results/3-订购/weekly_order.xlsx'))


def calculate_type_sum(data):
    type_column = data['供应商ID'].apply(lambda x: supply_type_dict[x])
    data['材料分类'] = type_column
    type_groups = data.groupby('材料分类')
    type_sum = []
    for name, group in type_groups:
        group.drop(columns='材料分类', inplace=True)
        type_sub_sum = group.iloc[:, 1:].sum().sum()
        type_sum.append(type_sub_sum)
    return type_sum


problem2_type_sum = calculate_type_sum(problem2_order)
problem3_type_sum = calculate_type_sum(problem3_order)

problem2_type_sum = problem2_type_sum / np.sum(problem2_type_sum)
problem3_type_sum = problem3_type_sum / np.sum(problem3_type_sum)

plt.rc('font', family='SimHei')
plt.rc('axes', unicode_minus=False)
plt.figure(figsize=(10, 6))
plt.pie(problem2_type_sum, labels=['A', 'B', 'C'], autopct='%1.1f%%', colors=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99'],
              wedgeprops={'edgecolor': 'black', 'linewidth': 2})
plt.title('2-各类别占比')
plt.savefig('./pictures/2-各类别占比.png', dpi=600)
plt.close()


plt.rc('font', family='SimHei')
plt.rc('axes', unicode_minus=False)
plt.figure(figsize=(10, 6))
plt.pie(problem3_type_sum, labels=['A', 'B', 'C'], autopct='%1.1f%%', colors=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99'],
              wedgeprops={'edgecolor': 'black', 'linewidth': 2})
plt.title('3-各类别占比')
plt.savefig('./pictures/3-各类别占比.png', dpi=600)
plt.close()