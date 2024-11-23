import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

financial_data = pd.read_csv('data/financial_data.csv')
LR = pd.read_csv('data/LR.csv')

ticker = financial_data.loc['TICKER_SYMBOL']
match_ticker = LR.loc[]
