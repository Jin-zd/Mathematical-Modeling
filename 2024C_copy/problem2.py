from problem3_data import *

# 单一目标函数
def target(x, Y, J, K, crop_sale_t, crop_land_to_yield_t, crop_price_t, crop_land_to_cost_t):
    x = x.reshape(dim_x, dim_y)
    sum_y = 0
    for k in K:
        sum_y += Y(x, k, J, crop_sale_t, crop_land_to_yield_t, crop_price_t)
    sum_jk = 0
    for j in J:
        for k in K:
            sum_jk += x[j_transform(j), k_transform(k)] * C(j, k, crop_land_to_cost_t) * S(j)
    return sum_y - sum_jk


I5 = range(34, 50)
J5 = range(66, 98)
K5 = range(37, 41)

dim_x, dim_y = len(J5), len(K5)

@ea.Problem.single
def func5(x, Y=Y2):
    x = x.reshape(dim_x, dim_y)
    f = 0
    for n in range(n_handle):
        f += target(x, Y, J5, K5, crop_sale_list_2030[n], crop_land_to_yield_list_2030[n], crop_price_list_2030[n],
                    crop_land_to_cost_copy_2029)
    f = f / n_handle

    cv = []
    for k in K5:
        sum_i = 0
        for i in I5:
            sum_i += M(i, k, x, J5)
        cv.append(sum_i - 4)

    sum_k = x.sum(axis=1)
    for x_s in sum_k:
        cv.append(x_s - 1)

    cv.append(f - 750000)
    return f, cv

res5 = multi_sega(dim_x, dim_y, func5, prophet=None)
result_to_excel(np.array(res5.get('Vars')).reshape(dim_x, dim_y), J5, K5, './results/2-2030/2030-食用菌-第二季.xlsx')