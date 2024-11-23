import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

data1 = pd.read_csv('data/BCHAIN-MKPRU.csv')
data2 = pd.read_csv('data/LBMA-GOLD.csv')

data21 = pd.DataFrame(data1['Date'], columns=['Date'])
data22 = pd.merge(data21, data2, on='Date', how='left')

data23 = data22.interpolate(method='linear', limit_direction='forward')
data23.iloc[0, 1] = data23.iloc[1, 1]
data23['USD (PM)'] = data23['USD (PM)']

scaler = StandardScaler()
data1['Value'] = scaler.fit_transform(data1['Value'].values.reshape(-1, 1))
data23['USD (PM)'] = scaler.fit_transform(data23['USD (PM)'].values.reshape(-1, 1))

result1 = adfuller(data1['Value'])
result2 = adfuller(data23['USD (PM)'])

with open('out/ADF_test_result.txt', 'w') as file:
    file.write('Gold:\n')
    file.write('ADF Statistic:' + str(result1[0]) + '\n')
    file.write('p-value:'+ str(result1[1]) + '\n')
    file.write('Critical Values:' + str(result1[4]) + '\n')
    file.write('Bchain:\n')
    file.write('ADF Statistic:' + str(result2[0]) + '\n')
    file.write('p-value:'+ str(result2[1]) + '\n')
    file.write('Critical Values:' + str(result2[4]) + '\n')

data23['diff'] = data23['USD (PM)'].diff()
data23['diff'] = data23['diff'].fillna(0)

data1['diff'] = data1['Value'].diff()
data1['diff']= data1['diff'].fillna(0)

result11 = adfuller(data1['diff'])
result21 = adfuller(data23['diff'])

with open('out/diff_ADF_test_result.txt', 'w') as file:
    file.write('Gold:\n')
    file.write('ADF Statistic:' + str(result11[0]) + '\n')
    file.write('p-value:'+ str(result11[1]) + '\n')
    file.write('Critical Values:' + str(result11[4]) + '\n')
    file.write('Bchain:\n')
    file.write('ADF Statistic:' + str(result21[0]) + '\n')
    file.write('p-value:'+ str(result21[1]) + '\n')
    file.write('Critical Values:' + str(result21[4]) + '\n')

acf_result1 = sm.tsa.acf(data1['diff'])
pacf_result1 = sm.tsa.pacf(data1['diff'])

acf_result2 = sm.tsa.acf(data23['diff'])
pacf_result2 = sm.tsa.pacf(data23['diff'])

# plot_acf(data1['diff'])
# plt.title('Gold ACF')
# plt.show()
#
# plot_pacf(data1['diff'])
# plt.title('Gold PACF')
# plt.show()
#
# plot_acf(data23['diff'])
# plt.title('Bchain ACF')
# plt.show()
#
# plot_pacf(data23['diff'])
# plt.title('Bchain PACF')
# plt.show()
