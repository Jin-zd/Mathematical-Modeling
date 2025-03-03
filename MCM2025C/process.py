import pandas as pd
import re

df = pd.read_csv('data/summerOly_athletes.csv', encoding='windows-1251')


def clean_team_name(team_name):
    return re.sub(r'-\d+$', '', team_name)


df['Team'] = df['Team'].apply(clean_team_name)
#
# df_sorted = df.sort_values(by='Year')
# medals = ['Gold', 'Silver', 'Bronze']
# df_medals = df_sorted[df_sorted['Medal'].isin(medals)]
#
# first_award_years = df_medals.groupby('Team')['Year'].min().reset_index()
# first_award_records = pd.merge(first_award_years, df_medals, on=['Team', 'Year'])
# first_award_records[first_award_records["Year"] >= 1952].to_csv('./data/player_first_medal_records.csv', index=False)

df_sorted = df.sort_values(by=["Team", "Event", "Year"])
first_participation = df_sorted.groupby(["Team", "Event"]).first().reset_index()
result = first_participation[first_participation["Medal"] != "No medal"]
result[result["Year"] >= 2000].to_csv('./data/first_get_medal_records.csv', index=False)
