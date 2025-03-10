import pandas as pd
import json

# Load the JSON data
with open('/home/shefi/Projects/samueli_ds/data/datasets/single_sample/metadata_sample.json') as f:
    data = json.load(f)

# Create the initial DataFrame
df = pd.DataFrame(data)

# Normalize the associated_entities field for each row and concatenate the results
df_entities = pd.json_normalize(df['associated_entities'].explode())
df = df.drop(columns=['associated_entities']).join(df_entities)

# ...existing code to work with df...
print(df)
