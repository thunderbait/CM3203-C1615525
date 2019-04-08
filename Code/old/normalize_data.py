import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize


# dataset understanding

# read CSV & flatten JSON fields

def load_df(csv_path='../input/train_v2.csv', nrows=None):
    JSON_COLS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    
    for column in JSON_COLS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df

#print(os.listdir("../input"))

df_small = load_df(nrows=20000)
df_train = load_df()
df_test = load_df("../input/test_v2.csv")
