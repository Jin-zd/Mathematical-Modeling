import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.model_selection import KFold, train_test_split

# 读取并处理数据
data2_1 = pd.DataFrame(pd.read_excel('data/铅钡未风化类.xlsx'))
data2_2 = pd.DataFrame(pd.read_excel('data/铅钡风化类.xlsx'))
data2_3 = pd.DataFrame(pd.read_excel('data/高钾未风化类.xlsx'))
data2_4 = pd.DataFrame(pd.read_excel('data/高钾风化类.xlsx'))
data2_5 = pd.DataFrame(pd.read_excel('data/铅钡类预测.xlsx'))
data2_6 = pd.DataFrame(pd.read_excel('data/高钾类预测.xlsx'))

data2_5 = data2_5.drop(data2_5.columns[0], axis=1)
data2_6 = data2_6.drop(data2_6.columns[0], axis=1)

figure_num = 1


# 新建文件夹
def make_dir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


# 提取数据
def extract(data, zero=True):
    indicators = []
    for i in range(data.shape[0] - 1):
        indicator = []
        for j in range(1, data.shape[1]):
            if zero:
                indicator.append(data.iloc[i, j])
            else:
                if data.iloc[i, j] == 0:
                    indicator.append(np.random.uniform(0.001, 0.003))
                else:
                    indicator.append(data.iloc[i, j])
        indicators.append(indicator)
    return indicators


# critic法计算权值
def critic(indicators):
    in_norm = indicators
    sigma = np.std(in_norm, axis=0)
    corr = np.corrcoef(in_norm.T)
    C = sigma * np.sum(1 - corr, axis=0)
    return C / np.sum(C)


e1 = np.array(extract(data2_1, False))
e2 = np.array(extract(data2_2, False))
e3 = np.array(extract(data2_3, False))
e4 = np.array(extract(data2_4, False))

data21a = pd.concat([data2_1, data2_5], axis=0)
data23a = pd.concat([data2_3, data2_6], axis=0)

m1 = data21a.iloc[:, [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13]].values
m2 = data23a.iloc[:, [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]].values

w1 = critic(np.concatenate([e1, e3]))
w2 = critic(np.concatenate(([e2, e4])))
w3 = critic(m1)
w4 = critic(m2)

make_dir('critic')
with open('critic/critic_result1.txt', 'w', encoding='utf-8') as file:
    file.write('各成分权值：\n')
    file.write('未风化: ' + str(w1) + '\n')
    file.write('风化: ' + str(w2) + '\n')

with open('critic/critic_result2.txt', 'w', encoding='utf-8') as file:
    file.write('各成分权值：\n')
    file.write('铅钡: ' + str(w3) + '\n')
    file.write('高钾: ' + str(w4))


# 绘制轮廓系数随聚类簇变化图
def plot_accuracy(S, path):
    global figure_num
    num = range(2, 11)

    plt.figure(figure_num)
    plt.xticks(num)
    plt.title('S')
    plt.plot(num, S, c='orange', label='S')
    plt.legend()
    plt.savefig(path + '/' + 'S.png')

    with open(path + '/' + 'S.txt', 'w') as file:
        file.write('S:' + str(S) + '\n')
    figure_num += 1


# 绘制K均值聚类的聚类簇图
def plot_KMeans(data, labels, center, path):
    global figure_num
    plt.figure(figure_num)
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50)
    plt.scatter(center[:, 0], center[:, 1], c='red', marker='x', s=100, label='Cluster Centers')
    plt.title('K-means')
    plt.legend()
    plt.savefig(path + '/best_k.png')
    figure_num += 1


# 选择最佳K均值聚类的聚类簇数
def KMeans_select(data, label):
    average = []
    best_k = None
    best_score = -1

    for k in range(2, 11):
        S = []
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(data):
            data_train, data_val = data[train_idx], data[val_idx]

            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(data_train)

            labels = kmeans.predict(data_val)
            flag = len(np.unique(labels))
            if 2 <= flag <= len(labels) - 1:
                scores = silhouette_score(data_val, labels)
                S.append(scores)
            else:
                S.append(0)

        avg = np.mean(S)
        average.append(avg)
        if avg > best_score:
            best_score = avg
            best_k = k

    plot_accuracy(average, label)
    return best_k


# K均值聚类
def K_Means(data, label):
    new_label = 'KMeans/' + label
    make_dir(new_label)

    data_train, data_test = train_test_split(data, random_state=42)
    best_k = KMeans_select(data_train, new_label)
    best_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    best_kmeans.fit(data_train)
    plot_KMeans(data_train, best_kmeans.labels_, best_kmeans.cluster_centers_, new_label)

    labels_test = best_kmeans.predict(data_test)
    s = silhouette_score(data_test, labels_test)
    chi = calinski_harabasz_score(data_test, labels_test)
    with open(new_label + '/best_KMeans.txt', 'w') as file:
        file.write('S:' + str(s) + '\n')
        file.write('CHI:' + str(chi))

    return KMeans(n_clusters=best_k, n_init=10).fit(data).labels_


# 选择层次聚类最佳聚类簇数
def HierarchicalClustering_select(data, label):
    average = []
    best_k = None
    best_score = -1

    for k in range(2, 11):
        S = []
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(data):
            data_train, data_val = data[train_idx], data[val_idx]

            agg = AgglomerativeClustering(n_clusters=k)
            agg = agg.fit(data_train)
            labels = agg.labels_

            flag = len(np.unique(labels))
            if 2 <= flag <= len(labels) - 1:
                scores = silhouette_score(data_train, labels)
                S.append(scores)
            else:
                S.append(0)

        avg = np.mean(S)
        average.append(avg)
        if avg > best_score:
            best_score = avg
            best_k = k

    plot_accuracy(average, label)
    return best_k


# 层次聚类
def hierarchical_clustering(data, label):
    new_label = 'HierarchicalClustering/' + label
    make_dir(new_label)

    data_train, data_test = train_test_split(data, random_state=42)
    best_k = HierarchicalClustering_select(data_train, new_label)
    best_agg = AgglomerativeClustering(n_clusters=best_k)
    best_agg = best_agg.fit(data_train)

    labels = best_agg.labels_
    s = np.mean(silhouette_score(data_train, labels))
    chi = np.mean(calinski_harabasz_score(data_train, labels))
    with open(new_label + '/best_HierarchicalClustering.txt', 'w') as file:
        file.write('S:' + str(s) + '\n')
        file.write('CHI:' + str(chi))

    return AgglomerativeClustering(n_clusters=best_k).fit(data).labels_


# 读取数据并储存聚类结果
data21b = data21a.iloc[:, [1, 3, 6, 9, 10]].values
data23b = data23a.iloc[:, [1, 3, 6, 9, 10]].values

all_labels1 = K_Means(data21b, '铅钡')
data21a['类别'] = all_labels1
all_labels2 = K_Means(data23b, '高钾')
data23a['类别'] = all_labels2

data21a.to_excel('KMeans/铅钡/铅钡.xlsx')
data23a.to_excel('KMeans/高钾/高钾.xlsx')

all_labels1 = hierarchical_clustering(data21b, '铅钡')
data21a['类别'] = all_labels1
all_labels2 = hierarchical_clustering(data23b, '高钾')
data23a['类别'] = all_labels2

data21a.to_excel('HierarchicalClustering/铅钡/铅钡.xlsx')
data23a.to_excel('HierarchicalClustering/高钾/高钾.xlsx')
