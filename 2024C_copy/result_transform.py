import os
from tools import *


def season_transform(result_matrix, season_num, data):
    if season_num == 1:
        result = result_matrix.iloc[:54, :].copy()
    else:
        result = result_matrix.iloc[54:82, :].copy()

    columns = data.columns
    for i in range(data.shape[0]):
        land = data.iloc[i, 0]
        basic_land = data.iloc[i, 1]
        for j in range(2, data.shape[1]):
            crop = columns[j]
            if data.iloc[i, j] == 1:
                result.loc[result['地块名'] == land, crop] = basic_area_list[basic_land - 1]

    if season_num == 1:
        result_matrix.iloc[:54, :] = result
    else:
        result_matrix.iloc[54:82, :] = result
    return result_matrix


def file_transform(file_path, result_matrix):
    data = pd.read_excel(file_path)
    if '第二季' in file_path:
        result_matrix = season_transform(result_matrix, 2, data)
    else:
        result_matrix = season_transform(result_matrix, 1, data)
    return result_matrix


def result_file_get(result_path, save_path, folder_path, year_num):
    result_matrix = pd.read_excel(result_path, sheet_name=year_num-2024)
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            result_matrix = file_transform(file_path, result_matrix)
    result_matrix.to_excel(save_path, index=False)


result_file_get('./data/附件3/result1_1.xlsx', './results/final/2024-result1_1.xlsx',
                './results/1-2024/0', 2024)
result_file_get('./data/附件3/result1_2.xlsx', './results/final/2024-result1_2.xlsx',
                './results/1-2024/50', 2024)
result_file_get('./data/附件3/result1_1.xlsx', './results/final/2025-result1_1.xlsx',
                './results/1-2025/0', 2025)
result_file_get('./data/附件3/result1_2.xlsx', './results/final/2025-result1_2.xlsx',
                './results/1-2025/50', 2025)

result_file_get('./data/附件3/result2.xlsx', './results/final/2024-result2.xlsx',
                './results/2-2024', 2024)
result_file_get('./data/附件3/result2.xlsx', './results/final/2025-result2.xlsx',
                './results/2-2025', 2025)
result_file_get('./data/附件3/result2.xlsx', './results/final/2026-result2.xlsx',
                './results/2-2026', 2026)
result_file_get('./data/附件3/result2.xlsx', './results/final/2027-result2.xlsx',
                './results/2-2027', 2027)
result_file_get('./data/附件3/result2.xlsx', './results/final/2028-result2.xlsx',
                './results/2-2028', 2028)
result_file_get('./data/附件3/result2.xlsx', './results/final/2029-result2.xlsx',
                './results/2-2029', 2029)
result_file_get('./data/附件3/result2.xlsx', './results/final/2030-result2.xlsx',
                './results/2-2030', 2030)