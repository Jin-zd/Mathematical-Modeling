import pandas as pd
from prophet import Prophet
# from neuralprophet import NeuralProphet
from sklearn.metrics import mean_squared_error

# plotting
import matplotlib.pyplot as plt

# settings
plt.style.use('seaborn')
plt.rcParams["figure.figsize"] = (16, 8)
df = pd.read_csv('./data/wp_log_peyton_manning.csv')
print(f'The dataset contains {len(df)} observations.')
df.head()

df.plot(x='ds', y='y', title='Log daily page views')

test_length = 365
df_train = df.iloc[:-test_length]
df_test = df.iloc[-test_length:]

prophet_model = Prophet()
prophet_model.fit(df_train)
future_df = prophet_model.make_future_dataframe(periods=test_length)
preds_df_1 = prophet_model.predict(future_df)
prophet_model.plot_components(preds_df_1)

prophet_model.plot(preds_df_1)

# nprophet_model = NeuralProphet()
# metrics = nprophet_model.fit(df_train, freq="D")
# future_df = nprophet_model.make_future_dataframe(df_train, periods = test_length, n_historic_predictions=len(df_train))
# preds_df_2 = nprophet_model.predict(future_df)
# nprophet_model.plot(preds_df_2)

# # prepping the DataFrame
# df_test['prophet'] = preds_df_1.iloc[-test_length:].loc[:, 'yhat']
# df_test['neural_prophet'] = preds_df_2.iloc[-test_length:].loc[:, 'yhat1']
# df_test.set_index('ds', inplace=True)
#
# print('MSE comparison ----')
# print(f"Prophet:\t{mean_squared_error(df_test['y'], preds_df_1.iloc[-test_length:]['yhat']):.4f}")
# print(f"NeuralProphet:\t{mean_squared_error(df_test['y'], preds_df_2.iloc[-test_length:]['yhat1']):.4f}")
#
# df_test.plot(title='Forecast evaluation')