import numpy as np
import pandas as pd
import geatpy as ea
from matplotlib import pyplot as plt


weekly_supply = pd.read_excel('data/附件1 近5年402家供应商的相关数据.xlsx', sheet_name=1)
weekly_mean = (weekly_supply.iloc[:, 2:].sum(axis=1) / weekly_supply.iloc[:, 2:].apply(lambda x: (x != 0).sum(), axis=1)).to_numpy()

loss_rate = pd.DataFrame(pd.read_excel('./data/附件2 近5年8家转运商的相关数据.xlsx'))
mean_loss_rate = loss_rate.iloc[:, 1:].sum().sum() / (loss_rate.iloc[:, 1:] != 0).astype(int).sum().sum() / 100

supply_type_dict = weekly_supply.set_index('供应商ID')['材料分类'].to_dict()
type_rate_dict = {'A': 0.6, 'B': 0.66, 'C': 0.72}

supple_id = weekly_supply['供应商ID']
supply_k = []
for i in range(len(supple_id)):
    supply_k.append(type_rate_dict[supply_type_dict[supple_id[i]]])
supply_k = np.tile(np.array(supply_k).reshape(-1, 1), (1, 24))

weekly_mean = np.tile(weekly_mean.reshape(-1, 1), (1, 24))
weekly_mean[weekly_mean > 6000] = 6000

pred_data = pd.DataFrame(pd.read_excel('./results/pred_weekly_supply.xlsx'))
pred_data = pred_data.rename(columns={pred_data.columns[0]: '供应商ID'})
weekly_mean = pd.DataFrame(weekly_mean)
weekly_mean.insert(0, '供应商ID', supple_id)


for i in range(weekly_mean.shape[0]):
    s_id = weekly_mean.iloc[i, 0]
    for k in range(pred_data.shape[0]):
        if s_id == pred_data.iloc[k, 0]:
            weekly_mean.iloc[i, 1:] = pred_data.iloc[k, 1:].to_numpy()
            break

weekly_mean = weekly_mean.iloc[:, 1:].to_numpy()
weekly_mean[weekly_mean < 0] = 0

n = 402 * 24

@ea.Problem.single
def func(x):
    x = x.reshape(402, 24)
    f = (x / supply_k * (1 - mean_loss_rate)).sum()
    cv = []
    for j in range(24):
        cv.append(x[:, j].sum() - 48000)
        cv.append(28200 * 24 * 1.2 - (x / supply_k).sum())
    return f, np.array(cv)


ub = []
lb = [0] * n

for i in range(weekly_mean.shape[0]):
    for j in range(weekly_mean.shape[1]):
        ub.append(weekly_mean[i, j])

problem = ea.Problem(
        name='problem',
        M=1,
        maxormins=[-1],
        Dim=n,
        varTypes=[0] * n,
        lb=lb,
        ub=ub,
        evalVars=func)

algorithm = ea.soea_SEGA_templet(
    problem,
    ea.Population(Encoding='RI', NIND=100),
    MAXGEN=10000,
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
order_res = pd.DataFrame(np.round(np.array(res.get('Vars')), 2).reshape(402, 24), columns=[f'week{i}' for i in range(1, 25)])
order_res.insert(0, '供应商ID', supple_id)
order_res.to_excel('./results/4-订购/weekly_order.xlsx', index=False)


def plot_line(x, y, xlabel=None, ylabel=None, title=None, save_path=None):
    plt.rc('font', family='SimHei')
    plt.rc('axes', unicode_minus=False)
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, c='#A07EE7',label='总产能')
    plt.plot(x, [28200] * len(x), linestyle='--', label='28200')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title)
    plt.savefig(save_path, dpi=600)
    plt.close()

weekly_supply = pd.read_excel('./data/附件1 近5年402家供应商的相关数据.xlsx', sheet_name=1)
supply_type_dict = weekly_supply.set_index('供应商ID')['材料分类'].to_dict()
type_rate_dict = {'A': 0.6, 'B': 0.66, 'C': 0.72}


loss_rate = pd.DataFrame(pd.read_excel('./data/附件2 近5年8家转运商的相关数据.xlsx'))
mean_loss_rate = loss_rate.iloc[:, 1:].sum().sum() / (loss_rate.iloc[:, 1:] != 0).astype(int).sum().sum() / 100

supply_k = []
for i in range(len(weekly_supply)):
    supply_k.append(type_rate_dict[supply_type_dict[weekly_supply.iloc[i, 0]]])
supply_k = np.tile(np.array(supply_k).reshape(-1, 1), (1, 24))


data = pd.DataFrame(pd.read_excel('./results/4-订购/weekly_order.xlsx')).iloc[:, 1:].to_numpy()
data = (data / supply_k).sum(axis=0)
plot_line(range(len(data)), data, xlabel='周数', ylabel='总产能', title='周总产能折线图', save_path='./pictures/weekly_order_sum.png')
print((data.sum() - 28200 * 24) / (28200 * 24))