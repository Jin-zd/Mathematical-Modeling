"""
problem1_result.py 生成展示前10种作物种植面积比例分布的饼状图。

数据从Excel文件中读取，生成的图表保存为PNG图像。
"""



import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

plt.rc('font', family='SimHei')
plt.rc('axes', unicode_minus=False)

# 最佳种植方案下，作物种植面积比例分布饼状图
def result_pie(data_path, title, save_path):
    data = pd.read_excel(data_path).iloc[: 82, 2:].fillna(0)
    data = data.sum().to_frame().reset_index()
    df = pd.DataFrame({'crop': data.iloc[:, 0], 'area': data.iloc[:, 1]})
    df_sorted = df.sort_values('area', ascending=False)
    top_10 = df_sorted.head(10)
    others = pd.DataFrame({'crop': ['其他'], 'area': [df_sorted.iloc[10:]['area'].sum()]})
    df_pie = pd.concat([top_10, others])
    colors = plt.cm.Spectral(np.linspace(0.1, 0.9, len(df_pie)))
    fig, ax = plt.subplots(figsize=(12, 8))
    wedges, texts, autotexts = ax.pie(df_pie['area'],
                                      labels=df_pie['crop'],
                                      colors=colors,
                                      autopct='%1.1f%%',
                                      pctdistance=0.85,
                                      labeldistance=1.05)
    plt.setp(texts, size=10)
    plt.setp(autotexts, size=9, weight="bold", color="white")
    ax.set_title(title, fontsize=16)
    ax.legend(wedges, df_pie['crop'],
              title="作物",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1),
              fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

result_pie('./results/final/2024-result1_1.xlsx', '2024作物种植面积比例分布（前10种）（滞销策略）',
           './pictures/2024作物种植面积比例分布（滞销策略）.png')
result_pie('./results/final/2024-result1_2.xlsx', '2024作物种植面积比例分布（前10种）（降价策略）',
              './pictures/2024作物种植面积比例分布（降价策略）.png')
result_pie('./results/final/2025-result1_1.xlsx', '2025作物种植面积比例分布（前10种）（滞销策略）',
                './pictures/2025作物种植面积比例分布（滞销策略）.png')
result_pie('./results/final/2025-result1_2.xlsx', '2025作物种植面积比例分布（前10种）（降价策略）',
                './pictures/2025作物种植面积比例分布（降价策略）.png')