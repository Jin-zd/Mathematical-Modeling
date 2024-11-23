"""
problem2_data.py 使用拉丁超立方采样生成多种群协同的遗传算法所需的数据。
主要功能包括：
- 生成拉丁超立方采样数据
- 生成多种群协同遗传算法所需的数据
"""



from tools import *


n_samples = 10
n_handle = 1

# 拉丁超立方采样
def dict_based_lhs(variable_ranges, n_samples):
    result = {}

    for var_name, (min_val, max_val) in variable_ranges.items():
        bins = np.linspace(min_val, max_val, n_samples + 1)

        samples = np.random.uniform(
            low=bins[:-1],
            high=bins[1:],
            size=n_samples
        )

        np.random.shuffle(samples)

        result[var_name] = samples

    return pd.DataFrame(result)

variable_dict = {
    '小麦和玉米销售量增长率': (0.05, 0.10),
    '其他作物销售量增长率': (-0.05, 0.05),
    '亩产量': (-0.10, 0.10),
    '食用菌售价降低率': (0.01, 0.05)
}

sampling_result = dict_based_lhs(variable_dict, n_samples)
sampling_result.to_excel('./results/采样结果.xlsx', index=False)
sampling_result = sampling_result.mean().to_frame().T


def samples_generator(n):
    crop_sale_list = []
    crop_land_to_yield_list = []
    crop_land_to_cost_copy = crop_land_to_cost.copy()
    crop_price_list = []
    for samples_num in range(n_handle):
        sample = sampling_result.iloc[samples_num]

        crop_sale_copy = crop_sale.copy()
        for crop in crop_sale_copy.keys():
            if crop == '小麦' or crop == '玉米':
                crop_sale_copy[crop] = crop_sale_copy[crop] * ((1 + sample['小麦和玉米销售量增长率']) ** n)
            else:
                crop_sale_copy[crop] = crop_sale_copy[crop] * ((1 + sample['其他作物销售量增长率']) ** n)
        crop_sale_list.append(crop_sale_copy)

        crop_land_to_yield_copy = crop_land_to_yield.copy()
        for outer_key in crop_land_to_yield_copy.keys():
            for inner_key in crop_land_to_yield_copy[outer_key].keys():
                crop_land_to_yield_copy[outer_key][inner_key] = crop_land_to_yield_copy[outer_key][inner_key] * (1 + sample['亩产量'])
        crop_land_to_yield_list.append(crop_land_to_yield_copy)

        for outer_key in crop_land_to_cost_copy.keys():
            for inner_key in crop_land_to_cost_copy[outer_key].keys():
                crop_land_to_cost_copy[outer_key][inner_key] = crop_land_to_cost_copy[outer_key][inner_key] * ((1 + 0.05) ** n)

        crop_price_copy = crop_price.copy()
        for crop in crop_price_copy.keys():
            crop_type_c = crop_type[crop]
            if crop == '羊肚菌':
                crop_price_copy[crop] = crop_price[crop] * ((1 - 0.05) ** n)
            elif crop_type_c == '食用菌':
                crop_price_copy[crop] = crop_price_copy[crop] * (1 - sample['食用菌售价降低率']) ** n
            elif crop_type_c == '蔬菜' or crop_type_c == '蔬菜（豆类）':
                crop_price_copy[crop] = crop_price_copy[crop] * ((1 + 0.05) ** n)
        crop_price_list.append(crop_price_copy)

    return crop_sale_list, crop_land_to_yield_list, crop_land_to_cost_copy, crop_price_list


crop_sale_list_2024, crop_land_to_yield_list_2024, crop_land_to_cost_copy_2024, crop_price_list_2024 = samples_generator(1)
crop_sale_list_2025, crop_land_to_yield_list_2025, crop_land_to_cost_copy_2025, crop_price_list_2025 = samples_generator(2)
crop_sale_list_2026, crop_land_to_yield_list_2026, crop_land_to_cost_copy_2026, crop_price_list_2026 = samples_generator(3)
crop_sale_list_2027, crop_land_to_yield_list_2027, crop_land_to_cost_copy_2027, crop_price_list_2027 = samples_generator(4)
crop_sale_list_2028, crop_land_to_yield_list_2028, crop_land_to_cost_copy_2028, crop_price_list_2028 = samples_generator(5)
crop_sale_list_2029, crop_land_to_yield_list_2029, crop_land_to_cost_copy_2029, crop_price_list_2029 = samples_generator(6)
crop_sale_list_2030, crop_land_to_yield_list_2030, crop_land_to_cost_copy_2030, crop_price_list_2030 = samples_generator(7)

