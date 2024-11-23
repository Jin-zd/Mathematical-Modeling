import matplotx
import numpy as np
import pandas as pd
import pyecharts.options as opts
from matplotlib import pyplot as plt
from pyecharts.charts import Radar


def weight_calculate(matrix):
    return_arr = [matrix[0][0] * 0.15 + matrix[0][1] * 45 + matrix[0][2] * 3.15,
                  matrix[1][0] * 3 + matrix[1][1] * 22.5 + matrix[1][2] * 9, matrix[2][0] * 3.75 + matrix[2][1] * 3.75,
                  matrix[3][0] * 7.5, matrix[4][0] * 4.5 + matrix[4][1] * 3]
    return return_arr


def features_calculate(data: pd.DataFrame) -> dict:
    p1 = data['player1'].iloc[1]
    p2 = data['player2'].iloc[1]
    p_dict = {p1: [0] * 11, p2: [0] * 11}

    server_num = {p1: 0, p2: 0}
    win_shot_num = {p1: 0, p2: 0}
    p1_server_width_dict = {'B': [], 'BC': [], 'BW': [], 'C': [], 'W': [], '0': []}
    p2_server_width_dict = {'B': [], 'BC': [], 'BW': [], 'C': [], 'W': [], '0': []}
    p1_server_depth_dict = {'CTL': [], 'NCTL': [], '0': []}
    p2_server_depth_dict = {'CTL': [], 'NCTL': [], '0': []}

    for i in range(len(data)):
        # 爆发力
        if data.iloc[i]['server'] == 1:
            p_dict[p1][0] += data.iloc[i]['speed_mph']
            server_num[p1] += 1
        else:
            p_dict[p2][0] += data.iloc[i]['speed_mph']
            server_num[p2] += 1
        # ace率
        p_dict[p1][1] += data.iloc[i]['p1_ace']
        p_dict[p2][1] += data.iloc[i]['p2_ace']
        # 反手接球率
        if data.iloc[i]['winner_shot_type'] != 0:
            if data.iloc[i]['p1_winner'] == 1:
                win_shot_num[p1] += 1
                if data.iloc[i]['winner_shot_type'] == 'B':
                    p_dict[p1][2] += 1
            else:
                win_shot_num[p2] += 1
                if data.iloc[i]['winner_shot_type'] == 'B':
                    p_dict[p2][2] += 1
        # 单次发球次数
        if data.iloc[i]['server'] == 1:
            p_dict[p1][3] += data.iloc[i]['serve_no']
        else:
            p_dict[p2][3] += data.iloc[i]['serve_no']
        # 双误率
        if data.iloc[i]['server'] == 1:
            p_dict[p1][4] += data.iloc[i]['p1_double_fault']
        else:
            p_dict[p2][4] += data.iloc[i]['p2_double_fault']
        # 非受迫性失误率
        p_dict[p1][5] = 0.5 - data['p1_unf_err'].sum() / len(data)
        p_dict[p2][5] = 0.5 - data['p2_unf_err'].sum() / len(data)
        # 策略能力
        if data.iloc[i]['server'] == 1 and data.iloc[i]['point_victor'] == 1:
            p1_server_width_dict[data.iloc[i]['serve_width']].append(1)
            p1_server_depth_dict[data.iloc[i]['serve_depth']].append(1)
        elif data.iloc[i]['server'] == 1 and data.iloc[i]['point_victor'] == 2:
            p1_server_width_dict[data.iloc[i]['serve_width']].append(0)
            p1_server_depth_dict[data.iloc[i]['serve_depth']].append(0)
        if data.iloc[i]['server'] == 2 and data.iloc[i]['point_victor'] == 2:
            p2_server_width_dict[data.iloc[i]['serve_width']].append(1)
            p2_server_depth_dict[data.iloc[i]['serve_depth']].append(1)
        elif data.iloc[i]['server'] == 2 and data.iloc[i]['point_victor'] == 1:
            p2_server_width_dict[data.iloc[i]['serve_width']].append(0)
            p2_server_depth_dict[data.iloc[i]['serve_depth']].append(0)

        # 掌控力
        p_dict[p1][8] = data['p1_break_pt_won'].sum() / data['p1_break_pt'].sum()
        p_dict[p2][8] = data['p2_break_pt_won'].sum() / data['p2_break_pt'].sum()
        # 网前掌控力
        p_dict[p1][9] = data['p1_net_pt_won'].sum() / data['p1_net_pt'].sum()
        p_dict[p2][9] = data['p2_net_pt_won'].sum() / data['p2_net_pt'].sum()

    del p1_server_width_dict['0']
    del p2_server_width_dict['0']
    del p1_server_depth_dict['0']
    del p2_server_depth_dict['0']

    p_dict[p1][0] = p_dict[p1][0] / server_num[p1]
    p_dict[p2][0] = p_dict[p2][0] / server_num[p2]
    p_dict[p1][1] = p_dict[p1][1] / server_num[p1]
    p_dict[p2][1] = p_dict[p2][1] / server_num[p2]
    p_dict[p1][2] = p_dict[p1][2] / win_shot_num[p1]
    p_dict[p2][2] = p_dict[p2][2] / win_shot_num[p2]
    p_dict[p1][3] = 2 - p_dict[p1][3] / server_num[p1]
    p_dict[p2][3] = 2 - p_dict[p2][3] / server_num[p2]
    p_dict[p1][4] = 0.2 - p_dict[p1][4] / server_num[p1]
    p_dict[p2][4] = 0.2 - p_dict[p2][4] / server_num[p2]
    p_dict[p1][6] = max(map(lambda x: x.count(1) / len(x), p1_server_width_dict.values()))
    p_dict[p2][6] = max(map(lambda x: x.count(1) / len(x), p2_server_width_dict.values()))
    p_dict[p1][7] = max(map(lambda x: x.count(1) / len(x), p1_server_depth_dict.values()))
    p_dict[p2][7] = max(map(lambda x: x.count(1) / len(x), p2_server_depth_dict.values()))
    p_dict[p1][10] = p_dict[p1][2]
    p_dict[p2][10] = p_dict[p2][2]

    return p_dict


def plot_radar(feature, player):
    feature = [feature]
    c = (
        Radar(init_opts=opts.InitOpts())
        .add_schema(
            schema=[
                opts.RadarIndicatorItem(name="Ex", max_=10),
                opts.RadarIndicatorItem(name="Pom", max_=10),
                opts.RadarIndicatorItem(name="Sc", max_=10),
                opts.RadarIndicatorItem(name="Con", max_=5),
                opts.RadarIndicatorItem(name="Tp", max_=5),
            ],
            splitarea_opt=opts.SplitAreaOpts(
                is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
            ),
            textstyle_opts=opts.TextStyleOpts(color="#000000", font_weight='bold', font_size=16),
        )
        .add(
            series_name=player,
            data=feature,
            areastyle_opts=opts.AreaStyleOpts(color="#FF0000", opacity=0.2),  # 区域面积，透明度
        )
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        .render(f"out/{player}_radar.html")
    )
    return c


def features_combin(data: pd.DataFrame):
    p1 = data['player1'].iloc[1]
    p2 = data['player2'].iloc[1]
    p_dict = features_calculate(data)

    p1_f_pre = [p_dict[p1][0: 3], p_dict[p1][3: 6], p_dict[p1][6: 8], [p_dict[p1][8]], p_dict[p1][9: 11]]
    p2_f_pre = [p_dict[p2][0: 3], p_dict[p2][3: 6], p_dict[p2][6: 8], [p_dict[p2][8]], p_dict[p2][9: 11]]
    p1_f = weight_calculate(p1_f_pre)
    p2_f = weight_calculate(p2_f_pre)
    return p1_f, p2_f


def other_features(data: pd.DataFrame, point: int):
    p1 = data['player1'].iloc[1]
    p2 = data['player2'].iloc[1]

    p_other_dict = {p1: [0] * 3, p2: [0] * 3}

    # 心理状态
    if point <= 5:
        victor = data.iloc[0: point]['point_victor']
    else:
        victor = data.iloc[point - 5: point]['point_victor']
    p_other_dict[p1][0] = (victor == 1).sum() / len(victor) * 10
    p_other_dict[p2][0] = (victor == 2).sum() / len(victor) * 10

    # 体力差距
    game = data[data['point_no'] == point]
    set_num = game['set_no'].values[0]
    set_match = data[data['set_no'] == set_num]
    match = set_match[set_match['point_no'] <= point]
    p1_dis = match['p1_distance_run'].sum()
    p2_dis = match['p2_distance_run'].sum()
    if p1_dis <= p2_dis:
        p_other_dict[p1][1] = 25
        rate = (p2_dis - p1_dis) / p1_dis
        p_other_dict[p2][1] = 25 * (1 - rate)
    else:
        p_other_dict[p2][1] = 25
        rate = (p1_dis - p2_dis) / p2_dis
        p_other_dict[p1][1] = 25 * (1 - rate)

    # 是否发球
    if game['server'].values[0] == 1:
        p_other_dict[p1][2] = 5
        p_other_dict[p2][2] = 0
    else:
        p_other_dict[p2][2] = 5
        p_other_dict[p1][2] = 0

    return p_other_dict


def plot_trend(data: pd.DataFrame, point: int):
    p1 = data['player1'].iloc[1]
    p2 = data['player2'].iloc[1]
    p1_score = []
    p2_score = []

    for i in range(60, point):
        data_in = data.iloc[: i + 1, :]
        p1_f, p2_f = features_combin(data_in)
        p_or = other_features(data_in, i)
        p1_score_t = sum(p1_f) + sum(p_or[p1])
        p2_score_t = sum(p2_f) + sum(p_or[p2])
        p1_score.append(p1_score_t)
        p2_score.append(p2_score_t)

    x = range(60, point)
    # with plt.style.context(matplotx.styles.pitaya_smoothie['light']):
    plt.figure()
    plt.plot(x, p1_score, label=p1, c='#A07EE7', linewidth=1)
    plt.plot(x, p2_score, label=p2, c='#3FEDCF', linewidth=1)
    plt.ylim([0, 100])
    plt.xlabel('points')
    plt.ylabel('scores')
    plt.legend()
    plt.savefig(f'out/{point-1}_trend.png', dpi=300)
    plt.close()

    return p1_score, p2_score
