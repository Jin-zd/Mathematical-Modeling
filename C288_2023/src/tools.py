import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from pyrepo_mcda.mcda_methods import CODAS, TOPSIS, WASPAS, VIKOR, SPOTIS, EDAS, MABAC, MULTIMOORA
from pyrepo_mcda.additions import rank_preferences
from pyrepo_mcda import normalizations as norms
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import IsolationForest
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, silhouette_score, calinski_harabasz_score
from sklearn.model_selection import train_test_split
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
from keras import *


# fisher 检验
def fisher_test(data):
    oddsratio, p_value = stats.fisher_exact(data)
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.values, annot=True, cmap="Blues", cbar=True)
    plt.show()
    return oddsratio, p_value


# ACF 自相关系数
def acf_test(data, lags):
    acf = pd.Series(data).autocorr(lags)
    plot_acf(pd.Series(data), lags=lags)
    return acf


# 乘法时间序列分解
def mul_time_series_decompose(data):
    decomposition = seasonal_decompose(pd.Series(data), model="multiplicative")
    decomposition.plot()
    plt.show()


# FP-Growth 分析
def fp_growth(data):
    te = TransactionEncoder()
    te_ary = te.fit(data).transform(data)
    frequent_items = fpgrowth(pd.DataFrame(te_ary, columns=te.columns_), min_support=0.6, use_colnames=True)
    rules = association_rules(frequent_items, metric="lift", min_threshold=1.0)
    return frequent_items, rules


# TSNE 降维
def tsne(X, y, n_components=2):
    t_sne = TSNE(n_components=n_components, perplexity=30, n_iter=1000, random_state=42)
    X_embedded = t_sne.fit_transform(X)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='viridis', s=50, alpha=0.6)
    plt.colorbar(scatter, label='Digit Label')
    plt.title('t-SNE Visualization of Dataset')
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    plt.show()

    return X_embedded


# OPLS-DA 模型
def opls_da(X, y, n_components=2):
    """
    OPLS-DA模型实现。

    参数:
    - X: 输入数据矩阵（特征矩阵）。
    - y: 输出标签向量（分类变量）。
    - n_components: PLS-DA模型的组件数量。

    返回:
    - X_ortho: 去除了与分类无关的系统变化后的数据。
    - X_pred: 与分类变量相关的预测特征数据。
    - pls_model: 训练好的PLS-DA模型。
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pls = PLSRegression(n_components=n_components)
    pls.fit(X_scaled, y)

    T = pls.x_scores_
    P = pls.x_loadings_
    W = pls.x_weights_

    T_ortho = T - np.dot(np.dot(T, W.T), W) / np.dot(W.T, W)
    P_ortho = np.dot(X_scaled.T, T_ortho) / np.dot(T_ortho.T, T_ortho)

    X_ortho = X_scaled - np.dot(T_ortho, P_ortho.T)
    X_pred = np.dot(T, P.T)

    return X_ortho, X_pred, pls


# Isolation Forest 异常检测
def isolation_forest(data, y, n_estimators=100, max_samples='auto', contamination='auto', random_state=None):
    """
    使用 Isolation Forest 进行异常检测

    参数:
    - data: 输入数据，通常是一个二维数组或 DataFrame
    - n_estimators: 树的数量，默认为 100
    - max_samples: 从数据中抽样用于训练每棵树的样本数，默认为 'auto'
    - contamination: 数据中异常点的比例，默认为 'auto'
    - random_state: 随机种子，用于保证结果的可重复性

    返回:
    - is_anomaly: 每个样本是否为异常点的布尔数组
    - anomaly_scores: 每个样本的异常分数，负值越小越可能是异常点
    """
    iso_forest = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        random_state=random_state
    )

    iso_forest.fit(data)

    # 预测是否为异常点（返回 -1 表示异常，1 表示正常）
    predictions = iso_forest.predict(data)
    is_anomaly = predictions == -1

    # 获取异常分数（负值，数值越小越可能是异常点）
    anomaly_scores = iso_forest.decision_function(data)

    # plt.figure(figsize=(10, 6))
    # # 使用 Seaborn 绘制散点图，异常点和正常点用不同的颜色表示
    # sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=is_anomaly,
    #                 palette={False: 'blue', True: 'red'}, s=50,
    #                 edgecolor='k', alpha=0.7)
    # plt.title('Isolation Forest Anomaly Detection')
    # plt.xlabel('Feature 1')
    # plt.ylabel('Feature 2')
    # plt.legend(['Normal', 'Anomaly'])
    # plt.show()

    disp = DecisionBoundaryDisplay.from_estimator(
        iso_forest,
        data,
        response_method="decision_function",
        alpha=0.5,
    )
    scatter = disp.ax_.scatter(data[:, 0], data[:, 1], c=y, s=20, edgecolor="k")
    handles, labels = scatter.legend_elements()
    disp.ax_.set_title("Path length decision boundary \nof IsolationForest")
    plt.axis("square")
    plt.legend(handles=handles, labels=["outliers", "inliers"], title="true class")
    plt.colorbar(disp.ax_.collections[1])
    plt.show()

    return is_anomaly, anomaly_scores

# 异常值箱线图
def boxplot_anomaly_detection(data):
    df = pd.DataFrame(data, columns=[f'S_{i + 1}' for i in range(data.shape[1])])

    df = df.T

    plt.figure(figsize=(15, 8))
    df.boxplot()
    plt.title('Boxplot of Different Days by Sample')
    plt.xlabel('Samples')
    plt.ylabel('Value')
    plt.xticks(rotation=90)
    plt.show()


# 核密度图
def kde_plot(data):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data, shade=True)
    plt.title('Kernel Density Estimation Plot')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.show()


# 相关系数柱状热力图
def plot_spearman_correlation(data, dependent_var):
    """
    Plots the Spearman correlation coefficients of independent variables with the dependent variable.

    Parameters:
    df (DataFrame): Pandas DataFrame containing the data.
    dependent_var (str): The name of the dependent variable.

    Returns:
    None
    """
    # Calculate Spearman correlation coefficients
    correlations = {}
    for column in data.columns:
        if column != dependent_var:
            coef, _ = spearmanr(data[column], data[dependent_var])
            correlations[column] = coef

    # Sort correlations
    sorted_correlations = dict(sorted(correlations.items(), key=lambda item: item[1]))

    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    colors = sns.color_palette("viridis", len(sorted_correlations))

    bar_plot = sns.barplot(
        x=list(sorted_correlations.values()),
        y=list(sorted_correlations.keys()),
        palette=colors,
        ax=ax
    )

    plt.xlabel("Spearman Correlation")
    plt.ylabel("Independent Variables")
    plt.title("Correlation Ranking of Independent Variables with Dependent Variable")

    norm = plt.Normalize(vmin=min(sorted_correlations.values()), vmax=max(sorted_correlations.values()))
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Correlation Strength")

    plt.show()


# 中心对数比变换
def clr_transform(X):
    """
    Perform Centered Log-Ratio (CLR) transformation on input data.

    Parameters:
    X (numpy.ndarray): Input matrix of shape (n, m), where n is the number of samples
                       and m is the number of features.

    Returns:
    numpy.ndarray: CLR-transformed matrix of the same shape as input.
    """
    # Add a small constant to avoid log(0)
    epsilon = np.finfo(float).eps
    X_positive = X + epsilon

    geometric_means = np.exp(np.mean(np.log(X_positive), axis=1))

    clr_matrix = np.log(X_positive / geometric_means[:, np.newaxis])

    return clr_matrix


# LSTM 长短期记忆网路
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)


def build_model(look_back):
    model = Sequential()
    model.add(layers.Input(shape=(look_back, 1)))
    model.add(layers.LSTM(50, return_sequences=False))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def evaluate_lookback(data, max_look_back):
    r2_scores = []

    for look_back in range(1, max_look_back + 1):
        X, Y = create_dataset(data, look_back)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        train_size = len(X) - 30
        X_train, X_test = X[:train_size], X[train_size:]
        Y_train, Y_test = Y[:train_size], Y[train_size:]

        model = build_model(look_back)
        model.fit(X_train, Y_train, epochs=300, batch_size=32, verbose=0)

        predictions = model.predict(X_test)
        r2 = r2_score(Y_test, predictions)
        r2_scores.append(r2)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_look_back + 1), r2_scores)
    plt.title('R-square vs Look Back Window Size')
    plt.xlabel('Look Back Window Size')
    plt.ylabel('R-square')
    plt.show()

    return r2_scores


def plot_lstm(data, pred_data):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(data)), data, c='blue', label='Actual')
    plt.plot(range(len(pred_data)), pred_data, c='orange', label='Predicted')
    plt.title('LSTM Result')
    plt.xlabel('Day')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()


def lstm(data: pd.Series, pred_date_num, max_look_back=30):
    data = data.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    r2_scores = evaluate_lookback(data_scaled, max_look_back)
    max_r2_score = max(r2_scores)
    best_look_back = r2_scores.index(max(r2_scores)) + 1

    X, Y = create_dataset(data_scaled, best_look_back)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    model = build_model(best_look_back)
    model.fit(X, Y, epochs=300, batch_size=32, verbose=1)
    raw_predictions = model.predict(X)
    raw_predictions = raw_predictions.flatten().tolist()

    last_back_days = data_scaled[-best_look_back:]
    predictions = []

    for i in range(pred_date_num):
        X_pred = last_back_days[-best_look_back:].reshape(1, best_look_back, 1)
        pred = model.predict(X_pred)
        predictions.append(pred[0, 0])
        last_back_days = np.append(last_back_days, pred)[-best_look_back:]

    predictions_rescaled = scaler.inverse_transform(np.array(raw_predictions + predictions).reshape(-1, 1))
    plot_lstm(data, predictions_rescaled)

    return max_r2_score, best_look_back, predictions_rescaled


# kmeans++
def kmeans(k_data):
    k_data = (k_data - np.mean(k_data, axis=0)) / np.std(k_data, axis=0)
    data_train, data_test = train_test_split(k_data, random_state=42)
    model = KMeans(n_clusters=4, random_state=21, init='k-means++', n_init='auto', max_iter=500)
    model.fit(data_train)
    labels_test = model.predict(data_test)
    s = silhouette_score(data_test, labels_test)
    chi = calinski_harabasz_score(data_test, labels_test)
    labels = model.predict(k_data)
    return labels, s, chi


# 综合评价
def evaluate(values, weights, criteria):
    vikor_model = VIKOR(normalization_method=norms.minmax_normalization)
    pref = vikor_model(values, weights, criteria)
    rank = rank_preferences(pref, reverse=False)
    return rank


def plot_regression_results(x, y_true, y_pred,
                            xlabel='X值', ylabel='Y值',
                            title='回归模型结果与95%置信区间'):
    """
    绘制回归模型结果，包括观测值、预测值和自动计算的95%置信区间。

    参数:
    x : array-like, 自变量值
    y_true : array-like, 实际观测值
    y_pred : array-like, 模型预测值
    xlabel : str, x轴标签
    ylabel : str, y轴标签
    title : str, 图表标题
    """

    x = np.array(x)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    residuals = y_true - y_pred
    residual_std = np.sqrt(np.sum(residuals ** 2) / (len(x) - 2))
    x_mean = np.mean(x)
    x_sq_sum = np.sum((x - x_mean) ** 2)

    se = residual_std * np.sqrt(1 + 1 / len(x) + (x - x_mean) ** 2 / x_sq_sum)

    # 计算95%置信区间
    df = len(x) - 2  # 自由度
    t_value = stats.t.ppf(0.975, df)  # 双尾95%置信区间的t值
    y_lower = y_pred - t_value * se
    y_upper = y_pred + t_value * se

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.rc('font', family='SimHei')
    plt.rc('axes', unicode_minus=False)
    plt.scatter(x, y_true, color='blue', alpha=0.5, label='实际数据')
    plt.plot(x, y_pred, color='red', label='预测值')
    plt.fill_between(x, y_lower, y_upper, color='lightblue', alpha=0.3, label='95% 置信区间')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


# 渐变色散点图
def gradient_scatterplot(x, y, c, cmap='viridis', s=50, alpha=0.6, title='Gradient Scatterplot'):
    """
    绘制渐变色散点图。

    参数:
    x : array-like, x轴数据
    y : array-like, y轴数据
    c : array-like, 颜色数据
    cmap : str, 颜色映射名称
    s : int, 散点大小
    alpha : float, 透明度
    title : str, 图表标题
    """

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(x, y, c=c, cmap=cmap, s=s, alpha=alpha)
    plt.colorbar(scatter, label='颜色值')
    plt.title(title)
    plt.xlabel('X轴')
    plt.ylabel('Y轴')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()
