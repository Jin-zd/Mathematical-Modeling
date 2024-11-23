import numpy as np
import pandas as pd
from scipy.optimize import minimize

rr = pd.read_csv('out/rr.csv')
np.set_printoptions(precision=64)


def ES_Daily(a, x):
    VaR = np.percentile(a, (1 - x) * 100)
    ES = a[a <= VaR].mean()
    return ES


cash_CVaR = 0

w = 0.5
cob = 0.02
cog = 0.01
x1, x2, x3 = [1000, 0, 0]

for i in range(0, len(rr)):
    Pbe = rr.iloc[i, 3]
    Pge = rr.iloc[i, 1]

    Pbf = rr.iloc[i, 2]
    Pgf = rr.iloc[i, 0]

    gold_CVaR = ES_Daily(rr.iloc[0: i + 1, 0].values, 0.05)
    bitcoin_CVaR = ES_Daily(rr.iloc[0: i + 1, 2].values, 0.05)


    def target(alpha):
        alpha1, alpha2, alpha3 = alpha
        global x1, x2, x3
        cap = x1 + x2 + x3
        com = abs(cap * alpha2 - x2) * cob + abs(cap * alpha3 - x3) * cog
        v = cap * alpha2 * (1 + Pbe) + cap * alpha3 * (1 + Pge) + cap * alpha1 - cap - com
        Q = - cash_CVaR * cap * alpha1 - bitcoin_CVaR * cap * alpha2 - gold_CVaR * cap * alpha3
        if com >= cap * alpha1:
            x1_p = 0
            x2_p = cap * alpha2 - (com - cap * alpha1) * (alpha2 / (alpha2 + alpha3))
            x3_p = cap * alpha3 - (com - cap * alpha1) * (alpha3 / (alpha2 + alpha3))
        else:
            x1_p = cap * alpha1 - com
            x2_p = cap * alpha2
            x3_p = cap * alpha3
        x1 = x1_p
        x2 = x2_p * (1 + Pbf)
        x3 = x3_p * (1 + Pgf)
        return w * Q - (1 - w) * v


    def constr(alpha):
        alpha1, alpha2, alpha3 = alpha
        return 1 - alpha1 - alpha2 - alpha3


    con = {'type': 'eq', 'fun': constr}
    bd = [(0, 1) for i in range(3)]
    res = minimize(target, np.array([0.8, 0.1, 0.1]), bounds=bd, constraints=con)
    with open('out/optimize.txt', 'a') as f:
        f.write(str(res.x) + '\n')
