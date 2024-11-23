import numpy as np
import pandas as pd
import emoji
import re

hair_dryer = pd.read_csv('data/hair_dryer.tsv', sep='\t')
microwave = pd.read_csv('data/microwave.tsv', sep='\t')
pacifier = pd.read_csv('data/pacifier.tsv', sep='\t')


def contains_emoji(text):
    if isinstance(text, float):
        text = str(text)
    return emoji.emoji_count(text) > 0


hair_dryer = hair_dryer[~hair_dryer['review_body'].apply(contains_emoji)]
microwave = microwave[~microwave['review_body'].apply(contains_emoji)]
pacifier = pacifier[~pacifier['review_body'].apply(contains_emoji)]


def remove_non_alphanumeric(text):
    if isinstance(text, float):
        text = str(text)
    cleaned_text = re.sub(r'[^a-zA-Z0-9]', '', text)
    return cleaned_text


hair_dryer_comments = hair_dryer['review_body']
microwave_comments = microwave['review_body']
pacifier_comments = pacifier['review_body']

hair_dryer_comments = hair_dryer_comments.apply(remove_non_alphanumeric)
microwave_comments = microwave_comments.apply(remove_non_alphanumeric)
pacifier_comments = pacifier_comments.apply(remove_non_alphanumeric)

