import pandas as pd
from problem2 import make_dir
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import precision_score

# 读取并处理数据
data2_1 = pd.DataFrame(pd.read_excel('data/铅钡未风化类.xlsx'))
data2_2 = pd.DataFrame(pd.read_excel('data/铅钡风化类.xlsx'))
data2_3 = pd.DataFrame(pd.read_excel('data/高钾未风化类.xlsx'))
data2_4 = pd.DataFrame(pd.read_excel('data/高钾风化类.xlsx'))

data3 = pd.DataFrame(pd.read_excel('data/预测数据.xlsx'))

data2_1['类型'] = [0] * data2_1.shape[0]
data2_2['类型'] = [0] * data2_2.shape[0]
data2_3['类型'] = [1] * data2_3.shape[0]
data2_4['类型'] = [1] * data2_4.shape[0]

data3_1 = pd.concat([data2_1, data2_3], axis=0)
data3_2 = pd.concat([data2_2, data2_4], axis=0)

X1 = data3_1.iloc[:, 1: 15]
Y1 = data3_1.iloc[:, 15]
X2 = data3_2.iloc[:, 1: 15]
Y2 = data3_2.iloc[:, 15]

data31 = data3.iloc[[0, 2, 3, 7], 2: 16]
data32 = data3.iloc[[1, 4, 5, 6], 2: 16]


# 使用神经网络进行分类预测
def classifier_pred(X, Y, data, label):
    make_dir(label)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(10,), activation='logistic', solver='lbfgs', max_iter=100)
    scores = cross_val_score(mlp, X_train, Y_train, cv=5)
    scores = scores.mean()
    mlp.fit(X_train, Y_train)

    Y_pred = mlp.predict(X_test)
    precision = precision_score(Y_test, Y_pred)

    with open(label + '/precision.txt', 'w') as file:
        file.write('score: ' + str(scores) + '\n')
        file.write('precision: ' + str(precision))

    return mlp.predict(data)


# 将预测结果写入excel表格
def classifier(X, Y, data_pred, data, label):
    pred = classifier_pred(X, Y, data_pred, label)
    count = 0
    for i in range(data.shape[0]):
        if data.iloc[i, 1] == label:
            if pred[count] == 1:
                data.iloc[i, 16] = '高钾'
            else:
                data.iloc[i, 16] = '铅钡'
            count += 1


classifier(X1, Y1, data31, data3, '无风化')
classifier(X2, Y2, data32, data3, '风化')
data3.to_excel('data/问题三预测结果.xlsx')
