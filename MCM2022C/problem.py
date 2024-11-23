import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from pycaret.time_series import *
from statsmodels.tsa.seasonal import seasonal_decompose

from data import gold_predict, bitcoin_predict

gold = pd.read_csv('data/LBMA-GOLD.csv')
bitcoin = pd.read_csv('data/BCHAIN-MKPRU.csv')

total_date = pd.DataFrame(bitcoin['Date'], columns=['Date'])
temp_data = pd.merge(total_date, gold, on='Date', how='left')

gold_inter = temp_data.interpolate(method='linear', limit_direction='forward')
gold_inter.iloc[0, 1] = gold_inter.iloc[1, 1]
gold_inter['USD (PM)'] = gold_inter['USD (PM)']

gold = gold_inter
gold['Date'] = range(1, len(gold) + 1)
bitcoin['Date'] = range(1, len(bitcoin) + 1)


# scaler = StandardScaler()
# gold['USD (PM)'] = scaler.fit_transform(gold['USD (PM)'].values.reshape(-1, 1))
# bitcoin['Value'] = scaler.fit_transform(bitcoin['Value'].values.reshape(-1, 1))
gold.to_csv('out/gold.csv', index=False)
bitcoin.to_csv('out/bitcoin.csv', index=False)

# setup(gold, target='USD (PM)', fold=5, session_id=123)
# best = create_model('arima')
# plot_model(best, plot='insample', save=True)
# plot_model(best, plot='diagnostics', save=True)

# setup(bitcoin, target='Value', fold=5, session_id=123)
# best = create_model('arima')
# plot_model(best, plot='insample', save=True)
# plot_model(best, plot='diagnostics', save=True)

gold_real_rr = gold['USD (PM)'].pct_change()
gold_real_rr = gold_real_rr[1:].reset_index()
gold_real_rr = gold_real_rr['USD (PM)']
gold_pre_rr = (gold_predict.iloc[1:, 0].values - gold.iloc[:-1, 1].values) / gold.iloc[:-1, 1].values
bitcoin_real_rr = bitcoin['Value'].pct_change()
bitcoin_real_rr = bitcoin_real_rr[1:].reset_index()
bitcoin_real_rr = bitcoin_real_rr['Value']
bitcoin_pre_rr = (bitcoin_predict.iloc[1:, 0].values - bitcoin.iloc[:-1, 1].values) / bitcoin.iloc[:-1, 1].values

rr = pd.DataFrame({'gold_real_rr': gold_real_rr, 'gold_pre_rr': gold_pre_rr, 'bitcoin_real_rr': bitcoin_real_rr,
                   'bitcoin_pre_rr': bitcoin_pre_rr})
rr.to_csv('out/rr.csv', index=False)





