import pandas as pd

data1 = pd.read_excel('data/ProblemCData.xlsx', sheet_name=0)
data2 = pd.read_excel('data/ProblemCData.xlsx', sheet_name=1)
data3 = pd.read_excel('data/Unit3.xlsx')

filtered_msn = data3[data3['Unit'].str.contains('Btu') == False]['MSN']
filtered_list = filtered_msn.tolist()
filtered_df = data1[~data1['MSN'].isin(filtered_list)]

grouped = data1.groupby('StateCode')

AZ = grouped.get_group('AZ')
CA = grouped.get_group('CA')
NM = grouped.get_group('NM')
TX = grouped.get_group('TX')

AZ = AZ.sort_values(by='Year', ascending=True)

# AZ['type_category'] = AZ['MSN'].str[:2]
# AZ_grouped = AZ.groupby('type_category')

# AZ_grouped_data = pd.ExcelWriter('out/AZ_grouped_data.xlsx')
# for group_name, group_df in AZ_grouped:
#     group_df.to_excel(AZ_grouped_data, sheet_name=group_name, index=False)
# AZ_grouped_data.save()
# CA = CA.sort_values(by='Year', ascending=True)

# data2['type_category'] = data2['MSN'].str[:2]
# data2_grouped = data2.groupby('type_category')
# data2_grouped_data = pd.ExcelWriter('out/data2_grouped_data.xlsx')
# for group_name, group_df in data2_grouped:
#     group_df.to_excel(data2_grouped_data, sheet_name=group_name, index=False)
# data2_grouped_data.save()


