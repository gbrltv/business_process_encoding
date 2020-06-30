import re
import os
import time
import pandas as pd
import numpy as np
from guppy import hpy; h=hpy()
from sklearn.feature_extraction.text import TfidfVectorizer

def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)

def read_log(path, log):
    """
    Reads event log and preprocess it
    """
    df_raw = pd.read_csv(f'{path}/{log}')
    df_raw['activity_name'] = df_raw['activity'].str.replace(' ', '')
    df_proc = df_raw[['case', 'activity_name', 'label']]
    del df_raw
    return df_proc

def extract_corpus(df):
    """
    Creates corpus for model training
    """
    cases, y, ids = [], [], []
    for group in df.groupby('case'):
        events = list(group[1].activity_name)
        cases.append(' '.join(x for x in events))
        y.append(list(group[1].label)[0])
        ids.append(list(group[1].case)[0])

    return ids, cases, y


path = 'event_logs'
save_path = 'encoding_results/tfidfvectorizer'
os.makedirs(save_path, exist_ok=True)
st = time.time()
for file in sorted_aphanumeric(os.listdir(path)):
    # reads event log
    df = read_log(path, file)

    # process cases and labels
    ids, cases, y = extract_corpus(df)

    start_memory = h.heap().size
    start_time = time.time()

    # generate model
    model = TfidfVectorizer()
    encoding = model.fit_transform(cases)

    end_time = time.time() - start_time
    end_memory = h.heap().size - start_memory

    # saving
    out_df = pd.DataFrame(encoding.toarray(), columns=[f'feature_{i}' for i in range(encoding.toarray().shape[1])])
    out_df['case'] = ids
    out_df['time'] = end_time
    out_df['memory'] = end_memory
    out_df['label'] = y
    out_df.to_csv(f'{save_path}/{file}', index=False)

    del out_df, cases, model, ids, encoding, y, df

    print(file, end_time, end_memory)
print('final time', time.time() - st)
