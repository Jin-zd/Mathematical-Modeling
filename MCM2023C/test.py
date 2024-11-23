import pandas as pd
from pycaret.regression import *
from main import try_nums, attribute

x = pd.DataFrame(attribute, columns=['vec1', 'vec2', 'vec3'])
y = pd.DataFrame(try_nums[:, 0], columns=['1try'])
df = pd.concat([x, y], axis=1)
df['1try'] = df['1try'].astype(float)

setup(data=df, target='1try', session_id=42)

catboost_model = create_model('lightgbm')

tuned_catboost_model = tune_model(catboost_model)

evaluate_model(tuned_catboost_model)
