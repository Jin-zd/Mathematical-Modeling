from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr
import pandas as pd


for i in range(0, 2):
    p1_data = pd.read_csv(f'out/set{i + 1}_p1.csv')
    p2_data = pd.read_csv(f'out/set{i + 1}_p2.csv')

    p1_score = p1_data['p1_score']
    p1_mom = p1_data['sub1_mom']
    p1_points = p1_data['p1_point']

    p2_score = p2_data['p2_score']
    p2_mom = p2_data['sub2_mom']
    p2_points = p2_data['p2_point']

    p1_person, p1_p = pearsonr(p1_points, p1_mom)
    p2_person, p2_p = pearsonr(p2_points, p2_mom)

    p1_spearman = spearmanr(p1_score, p1_mom)
    p2_spearman = spearmanr(p2_score, p2_mom)

    print(p1_person, p1_spearman)
    print(p2_person, p2_spearman)

    # # ADF检验
    # p1_score_diff = np.diff(p1_score)
    # p1_mom_diff = np.diff(p1_mom)
    #
    # p2_score_diff = np.diff(p2_score)
    # p2_mom_diff = np.diff(p2_mom)
    #
    # p1_result = sm.tsa.stattools.grangercausalitytests(np.column_stack((p1_score_diff, p1_mom_diff)), maxlag=1)
    # p2_result = sm.tsa.stattools.grangercausalitytests(np.column_stack((p2_score_diff, p2_mom_diff)), maxlag=1)
    #
    # with open(f'out/set{i + 1}_p1.txt', 'w') as f:
    #     f.write(f'p1_score_diff_test: {adfuller(p1_score_diff)[1]}\n\n')
    #     f.write(f'p1_mom_diff_test: {adfuller(p1_mom_diff)[1]}\n\n')
    #     f.write(f'p1_result: {p1_result}')
    #
    # with open(f'out/set{i + 1}_p2.txt', 'w') as f:
    #     f.write(f'p2_score_diff_test: {adfuller(p2_score_diff)[1]}\n\n')
    #     f.write(f'p2_mom_diff_test: {adfuller(p2_mom_diff)[1]}\n\n')
    #     f.write(f'p2_result: {p2_result}')
