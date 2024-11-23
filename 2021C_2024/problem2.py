import numpy as np
import pandas as pd
import geatpy as ea
from matplotlib import pyplot as plt

def plot_line(x, y, xlabel=None, ylabel=None, title=None, save_path=None):
    plt.rc('font', family='SimHei')
    plt.rc('axes', unicode_minus=False)
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, c='#A07EE7')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(save_path, dpi=600)
    plt.close()

# 读取数据
weekly_supply = pd.DataFrame(pd.read_excel('./results/final_supply.xlsx'))
weekly_order = pd.DataFrame(pd.read_excel('./results/final_order.xlsx'))

# 平均损耗率
loss_rate = pd.DataFrame(pd.read_excel('./data/附件2 近5年8家转运商的相关数据.xlsx'))
mean_loss_rate = loss_rate.iloc[:, 1:].sum().sum() / (loss_rate.iloc[:, 1:] != 0).astype(int).sum().sum() / 100

# 供应商类型
supply_type_dict = weekly_supply.set_index('供应商ID')['材料分类'].to_dict()
type_rate_dict = {'A': 0.6, 'B': 0.66, 'C': 0.72}

# 2-1 挑选最少的供应商
supple_id = weekly_supply['供应商ID']
supply_k = []
for i in range(len(supple_id)):
    supply_k.append(type_rate_dict[supply_type_dict[supple_id[i]]])
supply_k = np.array(supply_k)

v = pd.DataFrame(pd.read_excel('./results/pred_weekly_supply.xlsx')).iloc[:, 1:].to_numpy()
v[v < 0] = 0

n = 50

@ea.Problem.single
def func1(x, supply_k=supply_k, v=v):
    r = []
    for k in range(24):
        if k == 0:
            r.append(28200)
        else:
            v_k = v[:, k - 1].T
            r.append((1 - mean_loss_rate) * ((x * v_k) / supply_k).sum() - 28200 + r[k - 1])
    f = x.sum()
    cv = []
    for j in range(24):
        v_j = v[:, j].T
        cv.append(28200 - (1 - mean_loss_rate) * (x * (v_j / supply_k)).sum() - r[j])
        cv.append(-r[j])
    return f, np.array(cv)

ub = [1] * n
lb = [0] * n

problem3 = ea.Problem(
        name='problem3',
        M=1,
        maxormins=[1],
        Dim=n,
        varTypes=[1] * n,
        lb=lb,
        ub=ub,
        evalVars=func1)

algorithm = ea.soea_SEGA_templet(
    problem3,
    ea.Population(Encoding='BG', NIND=100),
    MAXGEN=1000,
    logTras=1,
    trappedValue=1e-7,
    maxTrappedCount=10)

res = ea.optimize(algorithm,
                  seed=42,
                  verbose=False,
                  drawing=1,
                  outputMsg=True,
                  drawLog=False,
                  saveFlag=False)
select_num = res.get('Vars')

# 2-2 订购方案
select_num = [1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1,
        1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
        1, 1, 1, 1, 1, 1]
weekly_supply['选择'] = select_num
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
v['选择'] = select_num
v = v[v['选择'] == 1]
v = v.drop(columns='选择')
pred = v.copy()
pred.iloc[:, 1:] = v.iloc[:, 1:].astype(int)
pred.to_excel('./results/select.xlsx', index=False)
v = v.iloc[:, 1:].to_numpy()
v[v < 0] = 0


@ea.Problem.single
def func2(x):
    x = x.reshape(42, 24)

    row_mean = np.nanmean(x, axis=1)
    nan_indices = np.where(np.isnan(x))
    x[nan_indices] = np.take(row_mean, nan_indices[0])

    f = (x * supply_m).sum()
    cv = []
    r = []
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

problem2 = ea.Problem(name='problem2',
                         M=1,
                         maxormins=[1],
                         Dim=n,
                         varTypes=[0] * n,
                         lb=lb,
                         ub=ub,
                         evalVars=func2)

algorithm = ea.soea_SEGA_templet(
    problem2,
    ea.Population(Encoding='RI', NIND=100),
    MAXGEN=4000,
    logTras=1,
    trappedValue=1e-7,
    maxTrappedCount=10)

res = ea.optimize(algorithm,
                  verbose=True,
                  drawing=1,
                  outputMsg=True,
                  drawLog=False,
                  saveFlag=False)
best_x = np.array(res.get('Vars')).reshape(42, 24)
pd.DataFrame(best_x.astype(int)).to_excel('./results/2-weekly_order.xlsx', index=False, header=False)
select_supply_id = weekly_supply['供应商ID'].reset_index(drop=True)
order_res = pd.DataFrame(best_x.astype(int), columns=[f'week{i + 1}' for i in range(24)])
order_res.insert(0, '供应商ID', select_supply_id)
order_res.to_excel('./results/2-订购/weekly_order.xlsx', index=False)

# 2-2 效果分析图
order_res = pd.DataFrame(pd.read_excel('./results/2-订购/weekly_order.xlsx'))
order_res = order_res.iloc[:, 1:].to_numpy()
s = []
for j in range(24):
    s.append((order_res[:, j].T / supply_k).sum() * (1 - mean_loss_rate))
plot_line(range(0, 24), s, '周数', '总产能', '周产能折线图', './pictures/product_sum.png')

r1 = []
for k in range(24):
    if k == 0:
        r1.append(28200)
    else:
        r1.append(r1[k - 1] + s[k - 1] - 28200)
plot_line(range(0, 24), r1, '周数', '库存', '周库存折线图', './pictures/save_sum.png')


weekly_supply = pd.read_excel('./data/附件1 近5年402家供应商的相关数据.xlsx', sheet_name=1)
data = weekly_supply.iloc[:, 2:].to_numpy()
supply_type_dict = weekly_supply.set_index('供应商ID')['材料分类'].to_dict()
type_rate_dict = {'A': 0.6, 'B': 0.66, 'C': 0.72}
supple_id = weekly_supply['供应商ID']
supply_k = []
for i in range(len(supple_id)):
    supply_k.append(type_rate_dict[supply_type_dict[supple_id[i]]])
supply_k = np.tile(np.array(supply_k).reshape(-1, 1), (1, 240))
data = ((data / supply_k) * (1 - mean_loss_rate)).sum(axis=0)
plot_line(range(len(data)), data, xlabel='周数', ylabel='总产能', title='周总产能折线图', save_path='./pictures/weekly_product_sum.png')
r2 = []
for k in range(240):
    if k == 0:
        r2.append(28200)
    else:
        r2.append(r2[k - 1] + data[k - 1] - 28200)
plot_line(range(len(r2)), r2, xlabel='周数', ylabel='总库存', title='周总库存折线图', save_path='./pictures/weekly_save_sum.png')

deviation = 28200 - r2[-1]
r2 = [x + deviation for x in r2]

plt.rc('font', family='SimHei')
plt.rc('axes', unicode_minus=False)
plt.figure(figsize=(6, 4))
plt.plot(range(0, 240), r2, c='#A07EE7')
plt.plot(range(240, 264), r1, c='orange')
plt.xlabel('周数')
plt.ylabel('总库存')
plt.savefig('./pictures/total_save_sum.png', dpi=600)
plt.close()


# 2-3 每周的转运方案
q = pd.DataFrame(pd.read_excel('./results/2-weekly_order.xlsx', header=None)).to_numpy()
l = np.array([0.01904769166666665, 0.009213704166666671, 0.0018605555555555539, 0.015704823529411764, 0.02889825301204818,
    0.005437611111111113, 0.02078833333333333, 0.010102827586206882]).T.reshape(8, 1)
row_means = np.mean(q, axis=1)

for i in range(q.shape[0]):
    q[i, q[i] > 6000] = row_means[i]

n = 42 * 8

for j in range(24):
    q_j = np.tile(q[:, j].reshape(42, 1), (1, 8))

    @ea.Problem.single
    def func3(x):
        x = x.reshape(42, 8)
        x_t = x.T
        f = np.trace(np.dot(q_j, x_t * l))
        cv = []
        for k in range(8):
            cv.append(np.dot(x_t, q_j)[k, k] - 6000)
        for k in range(42):
            cv.append(x[k, :].sum() - 1)
            cv.append(1 - x[k, :].sum())
        cv = np.array(cv)
        return f, cv

    ub = [1] * n
    lb = [0] * n

    problem3 = ea.Problem(
            name='problem3',
            M=1,
            maxormins=[1],
            Dim=n,
            varTypes=[1] * n,
            lb=lb,
            ub=ub,
            evalVars=func3)

    algorithm = ea.soea_SEGA_templet(
        problem3,
        ea.Population(Encoding='BG', NIND=100),
        MAXGEN=2000,
        logTras=1,
        trappedValue=1e-7,
        maxTrappedCount=10)

    res = ea.optimize(algorithm,
                      seed=42,
                      verbose=True,
                      drawing=1,
                      outputMsg=True,
                      drawLog=False,
                      saveFlag=False)
    result = np.array(res.get('Vars')).reshape(42, 8)
    pd.DataFrame({'供应商ID': weekly_supply['供应商ID'], 'T1': result[:, 0],
                  'T2': result[:, 1], 'T3':result[:, 2], 'T4': result[:, 3],
                  'T5': result[:, 4], 'T6': result[:, 5],
                  'T7': result[:, 6],  'T8': result[:, 7], }).to_excel(f'./results/2-转运/{j + 1}_week.xlsx', index=False)






