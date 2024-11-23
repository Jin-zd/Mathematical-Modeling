import geatpy as ea
import numpy as np
from scipy.optimize import minimize


def constr(x):
    con_sum = 0
    for i in range(100):
        con_sum += (100 - i) * x[i]
    return con_sum


def ineq(x):
    return [x[0] - 10,
            x[0] + 2 * x[1] - 20,
            x[0] + 2 * x[1] + 3 * x[2] - 30,
            x[0] + 2 * x[1] + 3 * x[2] + 4 * x[3] - 40,
            constr(x) - 1000]


@ea.Problem.single
def eval_vars(x):
    f = np.sum(x ** (1 / 2))
    cv = np.array(ineq(x))
    return f, cv


problem = ea.Problem(name='problem',
                     M=1, maxormins=[-1],
                     Dim=100,
                     varTypes=[0 for _ in range(100)],
                     lb=[0 for _ in range(100)],
                     ub=[10000 for _ in range(100)],
                     evalVars=eval_vars)
algorithm = ea.soea_SEGA_templet(problem,
                                 ea.Population(Encoding='RI', NIND=200),
                                 MAXGEN=1000,
                                 logTras=1,
                                 trappedValue=1e-6,
                                 maxTrappedCount=10)
res1 = ea.optimize(algorithm, seed=1, verbose=True, drawing=1, outputMsg=True, drawLog=False, saveFlag=True,
                   dirName='result')

obj = lambda x: np.sum(- (x ** (1 / 2)))


def constr1(x):
    return ineq(x)


con = {'type': 'ineq', 'fun': constr1}
bd = [(0, np.inf) for i in range(100)]
res2 = minimize(obj, np.random.randn(100), constraints=con, bounds=bd)
print(res2)
