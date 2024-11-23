from collections import Counter
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.cluster import KMeans

data = pd.read_excel("data/Problem_C_Data_Wordle.xlsx")


# 定义回归函数
def log_function(x, a, b, c):
    return a * np.log(x) + b * x + c


# 读取日期和尝试总次数
nums = data.iloc[2: 334, 4].tolist()
nums.reverse()
nums = np.array(nums)
date = np.array(range(1, 333))

# 拟合对数函数并作图
popt, pcov = curve_fit(log_function, date, nums)
plt.figure(num=1)
plt.bar(date, nums, label='original')
plt.plot(date, log_function(date, *popt), label='fit', color='red')
plt.legend()
plt.savefig('out/log_fit.png', dpi=300)

# 计算偏差和R平方，检验回归效果
predict_num = log_function(393, *popt)
error_diff = abs(nums - log_function(date, *popt)) / nums
RER = np.mean(error_diff)
R2 = r2_score(nums, log_function(date, *popt))

with open('out/results.txt', 'w') as file:
    file.write('总次数对数函数回归结果:\n')
    file.write(f"回归参数 a,b,c: {popt}\n")
    file.write(f"预测值: {predict_num}\n")
    file.write(f"R平方 (R^2): {R2}\n")

# 读取单词数据，计算各字母出现次数及百分比
words = data.iloc[1:, 3].copy()
words[21] = 'naive'
all_words = ''.join(words)
all_words = all_words.lower()
letter_freq = Counter(all_words)
del letter_freq[' ']
letter_freq_df = pd.DataFrame(list(letter_freq.items()), columns=['Letter', 'Frequency'])
count_sum = letter_freq_df['Frequency'].sum()
letter_freq_df['Percentage'] = (letter_freq_df['Frequency'] / count_sum).round(3)
letter_freq_df.to_excel('out/Letter_Freq.xlsx')

# 计算每个单词中各字母出现百分比值之和
fre_dict = letter_freq_df.set_index('Letter')['Percentage'].to_dict()
features_sum = []
for i in range(1, words.shape[0] + 1):
    per_sum = 0
    for j in range(len(words[i])):
        if words[i][j] == ' ':
            continue
        per_sum += fre_dict[words[i][j]]
    per_sum = np.round(per_sum, 3)
    features_sum.append(per_sum)
word_features = pd.concat([words.reset_index(drop=True), pd.DataFrame(features_sum)], axis=1, keys=['Word', 'Fre_sum'])

# 计算每个单词中字母个数
unique_Letter_Count = words.apply(lambda x: len(set(x)))
unique_Letter_Count = pd.concat([words, unique_Letter_Count], axis=1, keys=['Word', 'Unique_Letter_Count'])
unique_Letter_Count.to_excel('out/Unique_Letter_Count.xlsx')

# 计算各组合字母相连出现次数
columns = list('abcdefghijklmnopqrstuvwxyz')
index = list('abcdefghijklmnopqrstuvwxyz')
df = pd.DataFrame(0, index=index, columns=columns)
for i in range(1, words.shape[0] + 1):
    word = words[i].lower()
    word = word.replace(' ', '')
    for j in range(len(word) - 1):
        df.loc[word[j], word[j + 1]] += 1

# 绘制各个字母相连出现的次数的热力图
df.to_excel('out/Letter_Transition_Matrix.xlsx')
correlation_matrix = df.corr()
plt.figure(num=2, figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Heatmap')
plt.savefig('out/correlation_heatmap.png', dpi=300)

# 计算各字母组合的百分比
matrix_letter = np.array(df.iloc[1:, 1:])
letter_sum = np.sum(matrix_letter)
df = df / letter_sum

# 计算每个单词中出现的字母组合的百分比之和
combine_sum = []
for i in range(1, words.shape[0] + 1):
    temp_sum = 0
    word = words[i]
    word = word.replace(' ', '')
    for j in range(len(word) - 1):
        temp_sum += df.loc[word[j], word[j + 1]]
    temp_sum = np.round(temp_sum, 3)
    combine_sum.append(temp_sum)

word_features['combin_sum'] = combine_sum
word_features['Unique_Letter_Count'] = unique_Letter_Count['Unique_Letter_Count'].reset_index(drop=True)

# 导出单词属性
word_features.to_excel('out/Word_Features.xlsx')

# 计算困难模式百分比并使用对数函数拟合
Hp = data.iloc[1:, 5].reset_index(drop=True) / data.iloc[1:, 4].reset_index(drop=True)
Hp = np.array(Hp.values)
Hp = np.flip(Hp)
mask = Hp < 0.5
Hp_del = Hp[mask]

date_del = range(1, 359)
popt, pcov = curve_fit(log_function, date_del, Hp_del)
R2 = r2_score(Hp_del, log_function(date_del, *popt))

# 绘制拟合曲线并计算R平方
plt.figure(num=3)
plt.bar(date_del, Hp_del, label='original')
plt.plot(date_del, log_function(date_del, *popt), label='fit', color='red')
plt.legend()
plt.savefig('out/Hard_Model_Percentage.png', dpi=300)

with open('out/results.txt', 'a') as file:
    file.write('\n困难模式百分比回归结果:\n')
    file.write(f"回归参数 a,b,c: {popt}\n")
    file.write(f"R平方 (R^2): {R2}\n")

# 获取每个单词的属性，构建属性向量
date = range(1, 360)
Attr = np.ravel(np.reshape(log_function(date, *popt) - Hp, (1, -1))).astype(float)
feature1 = np.ravel(np.reshape(np.array(word_features.iloc[:, 1].values), (1, -1))).astype(float)
feature2 = np.ravel(np.reshape(np.array(word_features.iloc[:, 2].values), (1, -1))).astype(float)
feature3 = np.ravel(np.reshape(np.array(word_features.iloc[:, 3].values), (1, -1))).astype(float)

# 进行Person相关性检验和Spearman相关性检验
person_corr_result = [np.corrcoef(feature1, Attr)[0, 1], np.corrcoef(feature2, Attr)[0, 1],
                      np.corrcoef(feature3, Attr)[0, 1]]
spearman_corr_result = [spearmanr(feature1, Attr).correlation, spearmanr(feature2, Attr).correlation,
                        spearmanr(feature3, Attr).correlation]
with open('out/results.txt', 'a') as file:
    file.write('\nPerson相关系数:\n')
    file.write(f"相关性: {person_corr_result}\n")
    file.write('Spearman相关系数:\n')
    file.write(f"相关性: {spearman_corr_result}\n")

# 进行回归分析，使用多目标的梯度提升树
attribute = pd.concat([word_features.iloc[:, 1: 3] * 100, word_features.iloc[:, 3]], axis=1).values
try_nums = data.iloc[1:, 6: 13].values
X_train, X_test, y_train, y_test = train_test_split(attribute, try_nums, test_size=0.4, random_state=42)
reg_md = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=3, random_state=42))
reg_md = reg_md.fit(X_train, y_train)

# 计算回归结果的MSE和MAE
md_mse = mean_squared_error(y_test, reg_md.predict(X_test))
md_mae = mean_absolute_error(y_test, reg_md.predict(X_test))

# 计算目标单词的属性
target = 'eerie'
target_percentage = 0
target_combine = 0
for letter in target:
    target_percentage += fre_dict[letter]
for i in range(len(target) - 1):
    target_combine += df.loc[target[i], target[i + 1]]
target_feature = np.array([target_percentage, target_combine, 3])

# 目标单词预测结果
target_predict = reg_md.predict(target_feature.reshape(1, -1)).flatten()

with open('out/results.txt', 'a') as file:
    file.write('\n梯度提升树结果:\n')
    file.write(f'MSE: {md_mse}\n')
    file.write(f'MAE: {md_mae}\n')
    file.write(f'EERIE 的预测结果: {np.round(target_predict)}\n')

bar_width = 0.2
x = np.array(range(7))
x = x.astype(float)

# 生成高斯噪声，均值为0，方差为1
mean = 0
std_dev = 1
num_samples = 359
gaussian_noise = np.random.normal(mean, std_dev, num_samples)


# 定义敏感性分析函数
def sensitivity_test(feature_index):
    # 为数据添加高斯噪声
    attribute[:, feature_index] += gaussian_noise

    feature_train, feature_test, try_train, try_test = train_test_split(attribute, try_nums, test_size=0.4,
                                                                        random_state=42)
    md = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=3, random_state=42))
    md = md.fit(feature_train, try_train)
    noise_pred = md.predict(target_feature.reshape(1, -1)).flatten()

    # 绘制敏感性分析图
    fig, ax1 = plt.subplots(num=feature_index + 5, figsize=(8, 6))
    ax2 = ax1.twinx()
    ax1.bar(x, target_predict, color='blue', label='original', width=0.2)
    ax1.bar(x + bar_width, noise_pred, color='orange', label='noise', width=0.2)
    ax1.legend()
    xtick_positions = x + bar_width / 2
    xtick_labels = [f"{i} try" for i in range(1, 8)]
    plt.xticks(xtick_positions, xtick_labels)
    ax2.plot(x, (target_predict - noise_pred) / target_predict, color='grey', label='diff')
    ax2.set_ylim(-1, 1)
    ax2.legend(bbox_to_anchor=(0.952, 0.9))
    plt.savefig('out/noise' + str(feature_index) + '.png', dpi=300)


sensitivity_test(0)
sensitivity_test(1)
sensitivity_test(2)

# 获取聚类数据
vec1 = data.iloc[1:, 6: 9].sum(axis=1)
vec2 = data.iloc[1:, 9: 12].sum(axis=1)
vec3 = data.iloc[:, 12]
vec_df = pd.concat([vec1, vec2, vec3], axis=1, keys=['Column1', 'Column2', 'Column3'])
vec_df = vec_df.drop(vec_df.index[-1])
vector = np.array(vec_df.values)

# 进行K均值聚类
words_Kmeans = pd.concat([words, vec1, vec2, vec3], axis=1, keys=['Words', 'Vec1', 'Vec2', 'Vec3'])
words_Kmeans = words_Kmeans.drop(words_Kmeans.index[-1])
K_model = KMeans(n_clusters=5, random_state=42)
K_model.fit(vector)
center = K_model.cluster_centers_
labels = K_model.labels_

words_Kmeans['Cluster'] = labels
words_Kmeans.to_excel('out/Words_Kmeans.xlsx')

# 绘制聚类簇状图
fig = plt.figure(num=12)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(vector[:, 0], vector[:, 1], vector[:, 2], c=labels, marker='o', s=10)
ax.scatter(center[:, 0], center[:, 1], center[:, 2], c='r', marker='x', s=100)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(elev=20, azim=30)
plt.savefig('out/Kmeans.png', dpi=300)
