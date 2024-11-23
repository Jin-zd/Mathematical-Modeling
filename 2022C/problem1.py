import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# 读取数据
data1_1 = pd.DataFrame(pd.read_excel('data/表格一计数.xlsx', sheet_name=0))
data1_2 = pd.DataFrame(pd.read_excel('data/表格一计数.xlsx', sheet_name=1))
data1_3 = pd.DataFrame(pd.read_excel('data/表格一计数.xlsx', sheet_name=2))

data2_2 = pd.DataFrame(pd.read_excel('data/铅钡风化类.xlsx'))
data2_4 = pd.DataFrame(pd.read_excel('data/高钾风化类.xlsx'))


# 卡方检验
def chi2_test(data):
    array = []
    for i in range(data.shape[0] - 1):
        array_i = []
        for j in range(1, data.shape[1] - 1):
            array_i.append(data.iloc[i, j])
        array.append(array_i)
    res = chi2_contingency(np.array(array))
    return format(res.statistic, '.2f'), format(res.pvalue, '.2f')


with open('chi2_result.txt', 'w', encoding='utf-8') as file:
    file.write('chi2 Test Result:\n')
    file.write('纹饰: ' + str(chi2_test(data1_1)) + '\n')
    file.write('类型: ' + str(chi2_test(data1_2)) + '\n')
    file.write('颜色: ' + str(chi2_test(data1_3)))


# 线性映射
def convert(a, b, c, d, x):
    if b == a:
        return [(d - c) / 2] * len(x)
    else:
        return ((d - c) * x + (b * c - a * d)) / (b - a)


# 使用线性映射预测
def predict(data1, data2, label):
    df = data1.copy()
    for j in range(1, data1.shape[1]):
        b = data2.iloc[j - 1, 0]
        a = data2.iloc[j - 1, 1]
        d = data2.iloc[j - 1, 2]
        c = data2.iloc[j - 1, 3]
        x = data1.iloc[:, j].values
        y = convert(a, b, c, d, x)
        for k in range(len(y)):
            df.iloc[k, j] = format(y[k])
    df.to_excel('data/' + label + '预测.xlsx')


Mm1 = pd.DataFrame(pd.read_excel('data/铅钡类最大值最小值.xlsx'))
Mm2 = pd.DataFrame(pd.read_excel('data/高钾类最大值最小值.xlsx'))

predict(data2_2, Mm1, '铅钡类')
predict(data2_4, Mm2, '高钾类')
