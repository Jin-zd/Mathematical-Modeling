import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

comp123_in = pd.read_excel('./data/附件1：123家有信贷记录企业的相关数据.xlsx', sheet_name=1)

comp123_in_comp_groups = comp123_in.groupby('企业代号')

test_comp = comp123_in_comp_groups.get_group('E1')

test_date = test_comp['开票日期']
daily_group = test_comp.resample('D', on='开票日期')
weekly_group = test_comp.resample('W', on='开票日期')

money_sum = [week[1].iloc[:, 4].sum() for week in weekly_group]

plt.plot(money_sum)
plt.show()

