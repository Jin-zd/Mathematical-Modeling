import pandas as pd
from matplotlib import pyplot as plt
import tools as tl

data1 = pd.DataFrame(pd.read_excel("../data/附件1.xlsx"))
data2 = pd.DataFrame(pd.read_excel("../data/附件2.xlsx"))
data3 = pd.DataFrame(pd.read_excel("../data/附件3.xlsx"))
data4 = pd.DataFrame(pd.read_excel("../data/附件4.xlsx"))

data14 = pd.merge(data1, data4, on=["单品编码", "单品名称"], how="left")
data14.to_excel("../data_processed/附件14合并.xlsx")

# data23 = pd.merge(data2, data3, on="单品编码", how="left")
data23 = data2.copy()
data23["销售日期"] = pd.to_datetime(data23["销售日期"])
data23["扫码销售时间"] = pd.to_datetime(data23["扫码销售时间"])

sale_type = pd.get_dummies(data23["销售类型"])
has_discount = pd.get_dummies(data23["是否打折销售"])
cross_table = pd.crosstab(data23["销售类型"], data23["是否打折销售"])
oddsratio, p_value = tl.fisher_test(cross_table)
print(oddsratio, p_value)

# data23.to_excel("../data_processed/附件23合并.xlsx")

data23.set_index("销售日期", inplace=True)
daily_group = data23.resample('D')
weekly_group = data23.resample('W')
monthly_group = data23.resample('ME')
yearly_group = data23.resample('YE')


single_type_dict = data1.set_index("单品编码")["分类编码"].to_dict()


def sale_sum(merged_group, type_dict, type_num):
    type_sum = []
    for name, group in merged_group:
        sale = 0
        for i in range(group.shape[0]):
            curr_type_num = type_dict[group["单品编码"].iloc[i]]
            if curr_type_num == type_num:
                sale += group["销量(千克)"].iloc[i]
        type_sum.append(sale)
    return type_sum


product_types = {
    '花叶类': 1011010101,
    '花菜类': 1011010201,
    '水生根茎类': 1011010402,
    '茄类': 1011010501,
    '辣椒类': 1011010504,
    '食用菌类': 1011010801,
}

# 品类日销售量计算
type_daily_sales = {product: sale_sum(daily_group, single_type_dict, product_types[product]) for product in product_types}

# 品类周销售量计算
type_weekly_sales = {product: sale_sum(weekly_group, single_type_dict, product_types[product]) for product in product_types}

# 品类月销售量计算
type_monthly_sales = {product: sale_sum(monthly_group, single_type_dict, product_types[product]) for product in product_types}

# 品类年销售量计算
type_yearly_sales = {product: sale_sum(yearly_group, single_type_dict, product_types[product]) for product in product_types}

pd.DataFrame(type_daily_sales).to_excel("../data_processed/品类日销售量.xlsx", index=False)
pd.DataFrame(type_weekly_sales).to_excel("../data_processed/品类周销售量.xlsx", index=False)
pd.DataFrame(type_monthly_sales).to_excel("../data_processed/品类月销售量.xlsx", index=False)
pd.DataFrame(type_yearly_sales).to_excel("../data_processed/品类年销售量.xlsx", index=False)

'''
def plot_sale(sales, freq, type_name):
    with plt.style.context(matplotx.styles.pitaya_smoothie['light']):
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
        plt.plot(range(len(sales)), sales, c='#4682B4')
        plt.title(type_name)
        plt.savefig("../pictures/" + freq + "/" + type_name + ".png", dpi=400)
        plt.close()


for key in type_daily_sales.keys():
    plot_sale(type_daily_sales[key], "type_daily", key)

for key in type_weekly_sales.keys():
    plot_sale(type_weekly_sales[key], "type_weekly", key)

for key in type_monthly_sales.keys():
    plot_sale(type_monthly_sales[key], "type_monthly", key)

for key in type_yearly_sales.keys():
    plot_sale(type_yearly_sales[key], "type_yearly", key)

single_name_dict = data1.set_index("单品名称")["单品编码"].to_dict()


def single_sale_sum(merged_group, single_num):
    single_sum = []
    for name, group in merged_group:
        sale = 0
        for i in range(group.shape[0]):
            if group["单品编码"].iloc[i] == single_num:
                sale += group["销量(千克)"].iloc[i]
        single_sum.append(sale)
    return single_sum


# 单品日销售量计算
single_daily_sales = {product: single_sale_sum(daily_group, single_name_dict[product]) for product in single_name_dict.keys()}

# 单品周销售量计算
single_weekly_sales = {product: single_sale_sum(weekly_group, single_name_dict[product]) for product in single_name_dict.keys()}

# 单品月销售量计算
single_monthly_sales = {product: single_sale_sum(monthly_group, single_name_dict[product]) for product in single_name_dict.keys()}

# 单品年销售量计算
single_yearly_sales = {product: single_sale_sum(yearly_group, single_name_dict[product]) for product in single_name_dict.keys()}

for key in single_daily_sales.keys():
    plot_sale(single_daily_sales[key], "single_daily", key)

for key in single_weekly_sales.keys():
    plot_sale(single_weekly_sales[key], "single_weekly", key)

for key in single_monthly_sales.keys():
    plot_sale(single_monthly_sales[key], "single_monthly", key)

for key in single_yearly_sales.keys():
    plot_sale(single_yearly_sales[key], "single_yearly", key)
'''




