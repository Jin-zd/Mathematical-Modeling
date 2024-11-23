import pandas as pd
import networkx as nx
import geatpy as ea

graph1_data = pd.read_csv('./data/附件3.csv')
graph2_data = pd.read_csv('./data/附件4.csv')

G1 = nx.DiGraph()
G2 = nx.DiGraph()


def calculate_net_flow(G):
    G = nx.relabel_nodes(G, {n: str(n) for n in G.nodes()})

    net_flow = {str(node): 0 for node in G.nodes()}

    for u, v, data in G.edges(data=True):
        weight = data.get('weight', 1)
        net_flow[u] -= weight
        net_flow[v] += weight

    return net_flow

for _, row in graph1_data.iterrows():
    G1.add_edge(row.iloc[0], row.iloc[1], weight=row.iloc[2])

for _, row in graph2_data.iterrows():
    G2.add_edge(row.iloc[0], row.iloc[1])


G1_net_flow = calculate_net_flow(G1)

dim = G2.number_of_edges()

@ea.Problem.single
def func(x):
    G2.clear_edges()
    for index, row in graph2_data.iterrows():
        G2.add_edge(row.iloc[0], row.iloc[1], weight=x[index])
    G2_net_flow = calculate_net_flow(G2)
    f = sum(abs(G1_net_flow[key] - G2_net_flow[key]) for key in G1_net_flow if key in G2_net_flow)
    cv = []
    for key in G2_net_flow.keys():
        cv.append(abs(G1_net_flow[key] - G2_net_flow[key]) - 100)
    return f, cv


ub = [600] * dim
lb = [0] * dim

problem = ea.Problem(name='problem',
                     M=1,
                     maxormins=[1],
                     Dim=dim,
                     varTypes=[0] * dim,
                     lb=lb,
                     ub=ub,
                     evalVars=func)

algorithm = ea.soea_SEGA_templet(
    problem,
    ea.Population(Encoding='BG', NIND=100),
    MAXGEN=10000,
    logTras=1,
    trappedValue=1e-7,
    maxTrappedCount=10)

res = ea.optimize(algorithm,
                  verbose=False,
                  drawing=1,
                  outputMsg=True,
                  drawLog=False,
                  saveFlag=False)