import numpy as np
import pandas as pd
from problem2 import make_dir
from sklearn.decomposition import PCA

# 读取并处理数据
data4_1 = pd.DataFrame(pd.read_excel('data/铅钡未风化类.xlsx'))
data4_2 = pd.DataFrame(pd.read_excel('data/高钾未风化类.xlsx'))

data41 = data4_1.iloc[:, 1:]
data42 = data4_2.iloc[:, 1:]

make_dir('Relevance')

# 计算相关系数矩阵
pd.DataFrame(data41.corr()).to_excel('Relevance/铅钡相关系数.xlsx')
pd.DataFrame(data42.corr()).to_excel('Relevance/高钾相关系数.xlsx')


# 进行主成分分析
def PCA_method(data, label):
    pca = PCA(n_components=0.8).fit(data)

    np.set_printoptions(suppress=True)
    str_data1 = np.array2string(pca.explained_variance_ratio_, precision=6, separator=', ', suppress_small=True)
    str_data2 = np.array2string(pca.components_, precision=6, separator=', ', suppress_small=True)

    with open('Relevance/' + label + '.txt', 'w') as file:
        file.write('Contribution rate:\n')
        file.write(str_data1 + '\n')
        file.write('Coefficient matrix:\n')
        file.write(str_data2)


PCA_method(data41, '铅钡主成分分析')
PCA_method(data42, '高钾主成分分析')
