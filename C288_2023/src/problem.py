from tools import *
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# type_daily_sales = pd.read_excel('../data_processed/品类日销售量.xlsx').to_dict()
# type_weekly_sales = pd.read_excel('../data_processed/品类周销售量.xlsx').to_dict()

data = pd.read_excel('../data/附件.xlsx')
data.fillna('无', inplace=True)

le = LabelEncoder()
data['纹饰'] = le.fit_transform(data['纹饰'])
data['类型'] = le.fit_transform(data['类型'])
data['颜色'] = le.fit_transform(data['颜色'])
data['表面风化'] = le.fit_transform(data['表面风化'])
labels = data.iloc[:, 0].to_numpy()
features = data.iloc[:, 1:].to_numpy()


features = tsne(features, labels, n_components=2)
isolation_forest(features, labels)
