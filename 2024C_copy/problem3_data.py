from tools import *



def samples_generator(n, k=0.5):
    crop_land_to_cost_copy = crop_land_to_cost.copy()

    crop_sale_copy = crop_sale.copy()
    for crop in crop_sale_copy.keys():
        crop_type_c = crop_type[crop]
        if crop == '小麦' or crop == '玉米':
            crop_sale_copy[crop] = crop_sale_copy[crop] * ((1 + np.random.uniform(0.05, 0.1)) ** n)
        elif crop == '羊肚菌':
            crop_sale_copy[crop] = crop_sale_copy[crop] * ((1 + k * 0.05) ** n)
        elif crop_type_c == '食用菌':
            crop_sale_copy[crop] = crop_sale_copy[crop] * (1 + k * np.random.uniform(0.01, 0.05)) ** n
        elif crop_type_c == '蔬菜' or crop_type_c == '蔬菜（豆类）':
            crop_sale_copy[crop] = crop_sale_copy[crop] * ((1 - k * 0.05) ** n)


    crop_land_to_yield_copy = crop_land_to_yield.copy()
    for outer_key in crop_land_to_yield_copy.keys():
        for inner_key in crop_land_to_yield_copy[outer_key].keys():
            crop_land_to_yield_copy[outer_key][inner_key] = crop_land_to_yield_copy[outer_key][inner_key] * (1 + np.random.uniform(-0.1, 0.1))

    for outer_key in crop_land_to_cost_copy.keys():
        for inner_key in crop_land_to_cost_copy[outer_key].keys():
            crop_land_to_cost_copy[outer_key][inner_key] = crop_land_to_cost_copy[outer_key][inner_key] * ((1 + 0.05) ** n)

    crop_price_copy = crop_price.copy()
    for crop in crop_price_copy.keys():
        crop_type_c = crop_type[crop]
        if crop == '羊肚菌':
            crop_price_copy[crop] = crop_price[crop] * ((1 - 0.05) ** n)
        elif crop_type_c == '食用菌':
            crop_price_copy[crop] = crop_price_copy[crop] * (1 - np.random.uniform(0.01, 0.05)) ** n
        elif crop_type_c == '蔬菜' or crop_type_c == '蔬菜（豆类）':
            crop_price_copy[crop] = crop_price_copy[crop] * ((1 + 0.05) ** n)

    return crop_sale_copy, crop_land_to_yield_copy, crop_land_to_cost_copy, crop_price_copy


crop_sale_2024, crop_land_to_yield_2024, crop_land_to_cost_2024, crop_price_2024 = samples_generator(1)
crop_sale_2025, crop_land_to_yield_2025, crop_land_to_cost_2025, crop_price_2025 = samples_generator(2)
crop_sale_2026, crop_land_to_yield_2026, crop_land_to_cost_2026, crop_price_2026 = samples_generator(3)
crop_sale_2027, crop_land_to_yield_2027, crop_land_to_cost_2027, crop_price_2027 = samples_generator(4)
crop_sale_2028, crop_land_to_yield_2028, crop_land_to_cost_2028, crop_price_2028 = samples_generator(5)
crop_sale_2029, crop_land_to_yield_2029, crop_land_to_cost_2029, crop_price_2029 = samples_generator(6)
crop_sale_2030, crop_land_to_yield_2030, crop_land_to_cost_2030, crop_price_2030 = samples_generator(7)