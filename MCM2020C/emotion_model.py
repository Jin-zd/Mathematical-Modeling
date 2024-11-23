from data_processing import hair_dryer_comments, microwave_comments, pacifier_comments
import pandas as pd

train_data = pd.read_json("data/train.jsonl", lines=True)
test_data = pd.read_json("data/test.jsonl", lines=True)




