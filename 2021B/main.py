# python version： 3.10.11
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split


data1 = pd.DataFrame(pd.read_excel('附件1.xlsx'))


# # 提取温度和乙醇转化率数据
# # 若要提取C4烯烃选择性数据，将第18行的3改为5
# def extractXY(start):
#     x = []
#     y = []
#     for i in range(start, start + 5):
#         x.append(data1.iloc[i, 2])
#         y.append(data1.iloc[i, 3] / 100)
#     return np.array(x), np.array(y)
#
#
# def make_dir(path):
#     folder = os.path.exists(path)
#     if not folder:
#         os.makedirs(path)
#
#
# def polynomial(x, params):
#     y = 0
#     for i in range(len(params)):
#         y += params[i] * (x ** i)
#     return y
#
#
# def logistic(x, params):
#     p = polynomial(x, params)
#     return 1 / (1 + np.exp(-p))
#
#
# def draw(x, y, group, re1, re2):
#     plt.rc('font', family='SimHei')
#     plt.rc('font', size=16)
#     x0 = np.linspace(x[0], x[len(x) - 1], 1000)
#
#     plt.figure(figsize=(5, 10))
#     plt.subplot(211)
#     plt.title('二次多项式')
#     plt.ylim((0, 1))
#     plt.scatter(x, y, marker='x', color='red', s=40)
#     y1 = polynomial(x0, re1.params)
#     plt.plot(x0, y1, color='blue')
#     plt.grid()
#
#     plt.subplot(212)
#     plt.title('Logistic')
#     plt.ylim((0, 1))
#     plt.scatter(x, y, marker='x', color='red', s=40)
#     y2 = logistic(x0, re2.params)
#     plt.plot(x0, y2, color='orange')
#     plt.grid()
#     plt.savefig(group + '/' + group + '.png', dpi=500)
#
#
# def regression(group, start):
#     x, y = extractXY(start)
#     df1 = {'y': y, 'x': x}
#     df2 = {'y': np.log(y / (1 - y)), 'x': x}
#     re1 = sm.formula.ols('y ~ x + I(x ** 2)', df1).fit()
#     make_dir(group)
#     with open(group + '/' + group + '.txt', 'w') as file:
#         file.write(str(re1.summary()))
#     re2 = sm.formula.ols('y ~ x', df2).fit()
#     with open(group + '/' + group + '.txt', 'a') as file:
#         file.write('\n\n' + str(re2.summary()))
#     draw(x, y, group, re1, re2)
#
#
# # regression('A1', 0)
# # regression('A2', 5)
# # regression('A3', 10)

data2 = pd.DataFrame(pd.read_excel('附件2.xlsx'))
df = data2.iloc[2:, 1:].copy()
df.corr().to_excel('corr.xlsx')


data3 = data1.iloc[:, [2, 3, 10, 11, 12, 13]].copy()
X = data3.iloc[:, [0, 2, 3, 4, 5]].copy().values
Y = data3.iloc[:, [1]].copy().values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
mlp = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1000)
mlp.fit(X_train, Y_train)

Y_pred = mlp.predict(X_test).T
print("Predicted labels:", Y_pred)
print("True labels:", Y_test)
