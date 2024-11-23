import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
from scipy.stats import zscore
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
import sympy as sp
from sklearn.cross_decomposition import PLSRegression

data2 = pd.DataFrame(pd.read_excel('Data2.xls'))
data2 = data2.dropna(how='all')

aver = []


def average(start, end):
    data_copy = data2.iloc[start: end + 1, 2: 7].copy()
    single_aver = [data2.iloc[start, 1]]
    df_sum = data_copy.sum(axis=0)
    for j in range(0, 5):
        single_aver.append(df_sum[j] / (end - start + 1))
    aver.append(single_aver)


average(0, 4)
average(5, 7)
average(8, 11)
average(12, 14)
average(15, 17)
average(18, 20)
average(21, 24)

aver = pd.DataFrame(aver)
aver.to_excel('average.xlsx')

col_max = aver.max(axis=0)
col_min = aver.min(axis=0)

convert = []
ppm = []

for i in range(0, aver.shape[0]):
    ppm.append(aver.iloc[i, 0])
convert.append(ppm)

for j in range(1, 6):
    single_con = []
    for i in range(0, aver.shape[0]):
        single_con.append((aver.iloc[i, j] - aver.iloc[0, j]) / (col_max[j] - col_min[j]))
    convert.append(single_con)

convert = pd.DataFrame(np.array(convert).T)
convert.to_excel('convert.xlsx')


Y_train = [0, 30, 50, 80, 100]
X_train = []
Y_test = [20, 150]
X_test = []

for i in range(0, 7):
    single_x = []
    for j in range(1, 6):
        single_x.append(convert.iloc[i, j])
    if convert.iloc[i, 0] in Y_test:
        X_test.append(single_x)
    else:
        X_train.append(single_x)

mlp = MLPRegressor(hidden_layer_sizes=(20,), solver='lbfgs', max_iter=1000)
mlp.fit(X_train, Y_train)
Y_pred = mlp.predict(X_test).T


print(Y_pred)
print(Y_test)
print('R2：', r2_score(Y_test, Y_pred))
print('MSE:', mean_squared_error(Y_test, Y_pred))


# clf = DecisionTreeRegressor()
# clf = clf.fit(X_train, Y_train)
# Y_pred = clf.predict(X_test).T
# print(Y_pred)
# print(Y_test)
# print('R2：', r2_score(Y_test, Y_pred))
# print('MSE:', mean_squared_error(Y_test, Y_pred))
# print(clf.score(X_test, Y_test))
# print(clf.score(X_train, Y_train))

#
# df = {'y': np.array(Y_train), 'x': np.array(X_train)}
# re = sm.formula.ols('y~x', data=df).fit()
# print(re.summary())
#
# z_data = zscore(aver.iloc[:, 1: 6].values, ddof=1)
# pd.DataFrame(z_data).to_excel('z_data.xlsx')
# pca = PCA(n_components= 0.85).fit(z_data)
# rete = pca.explained_variance_ratio_
# com = pca.components_.T
# Z = pca.transform(z_data)
#
# df = {'y': np.array(ppm), 'x': np.array(Z)}
# re = sm.formula.ols('y~x', data=df).fit()
# print(re.summary())
# beta0 = re.params[0]
# z_beta = []
# for i in range(1, 3):
#     z_beta.append(re.params[i])
# beta = com @ np.array(z_beta).T
# R, G, B, H, S = sp.var('R, G, B, H, S')
# X = np.array([R, G, B, H, S])
# print('ppm = ', beta0 + X @ beta)
#
# ppm = zscore(ppm, ddof=1)
# pls = PLSRegression(n_components=5)
# pls.fit(z_data, ppm)
# Y_pred = pls.predict(z_data)
# print(ppm)
# print(Y_pred)
# print(r2_score(ppm, Y_pred))