import pandas as pd
from matplotlib import pyplot as plt

from lib import features_combin, plot_radar, other_features, plot_trend

features = pd.read_csv('data/Wimbledon_featured_matches.csv')
features = features.fillna(0)
features['speed_mph'] = features['speed_mph'].replace(0, 112)
features['speed_mph'] = features['speed_mph'] - 72
groups_label = []
label_num = 0
for i in range(len(features)):
    if features.iloc[i, 3] == '00:00:00':
        label_num += 1
        groups_label.append(label_num)
    else:
        groups_label.append(label_num)

features['groups_label'] = groups_label
features.to_csv('data/features.csv', index=False)
players_groups = features.groupby('groups_label')

# for i in range(1, 32):
#     data = players_groups.get_group(i).reset_index(drop=True)
#     x = data['point_no']
#     y1 = data['p1_points_won']
#     y2 = data['p2_points_won']
#
#     plt.figure()
#     plt.plot(x, y1, label=data['player1'].iloc[1], c='#A07EE7')
#     plt.plot(x, y2, label=data['player2'].iloc[1], c='#3FEDCF')
#     plt.xlabel('Points')
#     plt.ylabel('Points Won')
#     plt.legend()
#     plt.savefig(f'trend/points_{i}.png', dpi=300)

data = players_groups.get_group(31).reset_index(drop=True)
# p1_f, p2_f = features_combin(data)
# plot_radar(p1_f, data['player1'].iloc[1])
# plot_radar(p2_f, data['player2'].iloc[1])

point1 = 201
# point2 = 268
plot_trend(data, point1)
# plot_trend(data, point2)

# x = range(point2 + 2)
# p1_points = data['p1_points_won'].iloc[: point2 + 2]
# p2_points = data['p2_points_won'].iloc[: point2 + 2]
# # with plt.style.context(matplotx.styles.pitaya_smoothie['light']):
# plt.figure()
# plt.plot(x, p1_points, label=data['player1'].iloc[1], c='#A07EE7', linewidth=1)
# plt.plot(x, p2_points, label=data['player2'].iloc[1], c='#3FEDCF', linewidth=1)
# plt.xlabel('Points')
# plt.ylabel('Points Won')
# plt.legend()
# plt.savefig('out/points_won.png', dpi=300)
