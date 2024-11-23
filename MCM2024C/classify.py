import matplotx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pycaret.classification import *

class_data = pd.read_excel('data/labels.xlsx', sheet_name=0)
train_data = class_data.iloc[: -5]
pre_data = class_data.tail(5).iloc[:, 1:]
mean = 0
std_dev = 1

num_samples = len(train_data)
noise = np.random.normal(mean, std_dev, num_samples)
# train_data['rc'] = train_data['rc'] + noise

setup(train_data, target='label', session_id=123, fold=5)

model = create_model('xgboost')
print(model)

predictions = predict_model(model, data=pre_data)
print(predictions)

plot_model(model, plot='auc', save=True)
plot_model(model, plot='confusion_matrix', plot_kwargs={'percent': True}, save=True)
plot_model(model, plot='feature', save=True)
interpret_model(model, save=True)
