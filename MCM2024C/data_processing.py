import pandas as pd

features = pd.read_csv('data/Wimbledon_featured_matches.csv')

players = set(features['player1']).union(set(features['player2']))
print(sorted(players))
print(len(players))