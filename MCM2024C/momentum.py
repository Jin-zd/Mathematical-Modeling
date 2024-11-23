import matplotx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from lib import features_combin, other_features, plot_trend

features = pd.read_csv('data/features.csv')
atp = pd.read_csv('data/ATP.csv')


def event_calculate(data: pd.DataFrame, i: int, p: int):
    if p == 1:
        pr = 2
    else:
        pr = 1
    a1 = data.iloc[i]['server'] == p
    a2 = data.iloc[i]['server'] == pr

    s1 = data.iloc[i]['set_victor'] == p and a1
    s2 = data.iloc[i]['point_victor'] == p and s1
    s3 = data.iloc[i]['point_victor'] == p and data.iloc[i]['serve_no'] == pr and a1
    s4 = data.iloc[i][f'p{pr}_break_pt_missed'] == 1 and a1
    s5 = data.iloc[i]['game_victor'] == p and a1
    s6 = data.iloc[i]['point_victor'] == p and a1
    s7 = data.iloc[i][f'p{p}_ace'] == 1 and a1
    s8 = data.iloc[i][f'p{p}_winner'] == 1 and a1
    s9 = data.iloc[i][f'p{p}_net_pt_won'] == 1 and a1
    r1 = data.iloc[i]['serve_no'] == p and data.iloc[i]['point_victor'] == p and a2
    r2 = data.iloc[i]['serve_no'] == pr and data.iloc[i]['point_victor'] == p and a2
    r3 = data.iloc[i][f'p{p}_break_pt_won'] == 1 and a2
    r4 = data.iloc[i]['game_victor'] == p and a2
    r5 = data.iloc[i]['set_victor'] == p and a2
    r6 = data.iloc[i][f'p{p}_net_pt_won'] == 1 and a2
    r7 = data.iloc[i][f'p{pr}_winner'] == 1 and a2

    s_1 = data.iloc[i][f'p{p}_unf_err'] == 1 and a1
    s_2 = data.iloc[i][f'p{p}_double_fault'] == 1 and a1
    r_1 = data.iloc[i][f'p{p}_unf_err'] == 1 and a2

    if a1:
        pi = sum([s1, s2, s4, s5, s7, s8, s9])
        ni = sum([s_1, s_2])
    else:
        pi = sum([r1, r2, r3, r4, r6, r7])
        ni = sum([r_1])
    return pi - ni


def self_exp_calculate(data: pd.DataFrame, i: int, p: int):
    if p == 1:
        diff_game = data.iloc[i]['p1_games'] - data.iloc[i]['p2_games']
        diff_set = data.iloc[i]['p1_sets'] - data.iloc[i]['p2_sets']
        diff_points = data.iloc[i]['p1_points_won'] - data.iloc[i]['p2_points_won']
    else:
        diff_game = data.iloc[i]['p2_games'] - data.iloc[i]['p1_games']
        diff_set = data.iloc[i]['p2_sets'] - data.iloc[i]['p1_sets']
        diff_points = data.iloc[i]['p2_points_won'] - data.iloc[i]['p1_points_won']
    return 3 * diff_game + 5 * diff_set + diff_points


w1 = 0.2
w2 = 0.0002

momentum1_zip = []
momentum1_zip_up = []
momentum1_zip_down = []

players_groups = features.groupby('groups_label')

for name, group in players_groups:
    group = group.reset_index(drop=True)
    # group = players_groups.get_group(31)
    p1 = group.iloc[1]['player1']
    p2 = group.iloc[1]['player2']
    exp1 = []
    exp2 = []
    # exp1_up = []
    # exp2_up = []
    # exp1_down = []
    # exp2_down = []
    sub_momentum1 = []
    sub_momentum1_up = []
    sub_momentum1_down = []
    for i in range(len(group)):
        if i == 0:
            exp1.append(atp[atp['Players'] == p1].iloc[:, 1: 7].mean(axis=1).values[0])
            exp2.append(atp[atp['Players'] == p2].iloc[:, 1: 7].mean(axis=1).values[0])
            # exp1_up.append(exp1[0])
            # exp2_up.append(exp2[0])
        pni1 = event_calculate(group, i, 1)
        pni2 = event_calculate(group, i, 2)
        self_exp1 = self_exp_calculate(group, i, 1)
        self_exp2 = self_exp_calculate(group, i, 2)
        exp1.append(exp1[i - 1] * np.sqrt((1 + w1 * pni1)) + w2 * self_exp1)
        exp2.append(exp2[i - 1] * np.sqrt((1 + w1 * pni2)) + w2 * self_exp2)
        # exp1_up.append(exp1[i - 1] * np.sqrt((1 + w1 * 2 * pni1)) + w2 * self_exp1)
        # exp2_up.append(exp2[i - 1] * np.sqrt((1 + w1 * 2 * pni2)) + w2 * self_exp2)
        # exp1_down.append(exp1[i - 1] * np.sqrt((1 + w1 * 0.5 * pni1)) + w2 * self_exp1)
        # exp2_down.append(exp2[i - 1] * np.sqrt((1 + w1 * 0.5 * pni2)) + w2 * self_exp2)
        # sub_momentum1.append(exp1[i] / (exp1[i] + exp2[i]))
        # sub_momentum1_up.append(exp1_up[i] / (exp1_up[i] + exp2_up[i]))
        # sub_momentum1_down.append(exp1_down[i] / (exp1_down[i] + exp2_down[i]))
    momentum1_zip.append(sub_momentum1)
# momentum1_zip_up.append(sub_momentum1_up)
# momentum1_zip_down.append(sub_momentum1_down)

momentum1 = [item for sublist in momentum1_zip for item in sublist]
momentum1_up = [item for sublist in momentum1_zip_up for item in sublist]
momentum1_down = [item for sublist in momentum1_zip_down for item in sublist]
momentum2 = [1 - item for item in momentum1]
# momentum = pd.DataFrame({'momentum1': momentum1, 'momentum2': momentum2})
# momentum.to_csv('out/momentum.csv', index=False)
momentum_last = momentum1_zip[-1]
op_momentum_last = [1 - item for item in momentum_last]

# with plt.style.context(matplotx.styles.pitaya_smoothie['light']):
#     x = range(len(momentum_last))
#     plt.figure(figsize=(16, 8))
#     plt.plot(x, momentum_last, label=r'$\eta = 0.2$')
#     plt.plot(x, momentum1_up, label=r'$\eta = 0.4$')
#     plt.plot(x, momentum1_down, label=r'$\eta = 0.1$')
#     plt.ylim((0, 1))
#     # plt.axhline(y=0, color='black')
#     # plt.plot(x, op_momentum_last, label='Momentum2')
#     plt.xlabel('Points')
#     plt.ylabel('Down Percent')
#     plt.legend()
#     plt.savefig('out/momentum_up.png', dpi=300)

# for i in range(1, 3):
i = 31
data = players_groups.get_group(i).reset_index(drop=True)
p1 = data['player1'].iloc[1]
p2 = data['player2'].iloc[1]

p1_point = data['p1_points_won']
p2_point = data['p2_points_won']

p1_score, p2_score = plot_trend(data, len(data) + 1)

sub1_mom = momentum1_zip[i - 1]
sub2_mom = [1 - item for item in sub1_mom]

start_index1 = len(sub1_mom) - len(p1_score)
sub1_mom = sub1_mom[start_index1:]
p1_point_t = p1_point[start_index1:]
start_index2 = len(sub2_mom) - len(p2_score)
sub2_mom = sub2_mom[start_index2:]
p2_point_t = p2_point[start_index2:]

pd.DataFrame({'p1_score': p1_score, 'sub1_mom': sub1_mom, 'p1_point': p1_point_t}).to_csv(f'out/set{i}_p1.csv',
                                                                                          index=False)
pd.DataFrame({'p2_score': p2_score, 'sub2_mom': sub2_mom, 'p2_point': p2_point_t}).to_csv(f'out/set{i}_p2.csv',
                                                                                          index=False)
