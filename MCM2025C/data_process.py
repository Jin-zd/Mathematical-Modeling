import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

athletes = pd.read_csv('data/summerOly_athletes.csv')
athletes_2000 = athletes[athletes['Year'] >= 2000].sort_values(by=["Year", "Team"], ascending=True)
athletes_2000.to_csv('data/summerOly_athletes_2000.csv', index=False)

medal_counts = pd.read_csv('data/summerOly_medal_counts.csv')
medal_counts_2000 = medal_counts[medal_counts['Year'] >= 2000].sort_values(by=["Year", "Rank"], ascending=True)
medal_counts_2000.to_csv('data/summerOly_medal_count_2000.csv', index=False)

programs = pd.read_csv('data/summerOly_programs.csv', encoding='windows-1251')
programs = programs.head(len(programs) - 5)

years = programs.columns[4:]

for _, row in programs.iterrows():
    sport = row['Sport']
    discipline = row['Discipline']
    counts = row[4:]

    counts = pd.to_numeric(counts, errors='coerce').fillna(0).astype(int)

    plt.figure(figsize=(10, 6))
    plt.plot(years, counts, marker='o', label=f'{sport} - {discipline}')
    plt.title(f'{sport} - {discipline} programs numbers change')
    plt.xlabel('Year')
    plt.ylabel('Programs')
    plt.xticks(rotation=45)
    plt.ylim([0, counts.max() + 1])
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'images/programs_change/{sport}_{discipline}_programs_numbers_change.png')
    plt.close()