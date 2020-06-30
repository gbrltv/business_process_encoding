import re
import os
import time
import pandas as pd
import numpy as np
from guppy import hpy; h=hpy()
from gensim.models import Word2Vec

def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)

def read_log(path, log):
    """
    Reads event log and preprocess it
    """
    df_raw = pd.read_csv(f'{path}/{log}')
    df_raw['activity_name'] = df_raw['activity'].str.replace(' ', '-')
    # labels = [1 if x == 'normal' else -1 for x in df_raw['label']]
    # df_raw['label'] = labels
    df_proc = df_raw[['case', 'activity_name', 'label']]
    del df_raw
    return df_proc

def cases_y_list(df):
    """
    Creates a list of cases for model training
    """
    cases, y, ids = [], [], []
    for group in df.groupby('case'):
        events = list(group[1].activity_name)
        cases.append([''.join(x) for x in events])
        y.append(list(group[1].label)[0])
        ids.append(list(group[1].case)[0])

    return ids, cases, y

def create_model(cases, size, window, min_count):
    """
    Creates a word2vec model
    """
    model = Word2Vec(size=size, window=window, min_count=min_count, workers=-1)
    model.build_vocab(cases)
    model.train(cases, total_examples=len(cases), epochs=10)

    return model

def average_feature_vector(cases, model):
    """
    Computes average feature vector for each trace
    """
    vectors = []
    for case in cases:
        case_vector = []
        for token in case:
            try:
                case_vector.append(model.wv[token])
            except KeyError:
                pass
        vectors.append(np.array(case_vector).mean(axis=0))

    return vectors

path = 'event_logs'
save_path = 'encoding_results/word2vec'
os.makedirs(save_path, exist_ok=True)
st = time.time()
for file in sorted_aphanumeric(os.listdir(path)):
    # reads event log
    df = read_log(path, file)

    # process cases and labels
    ids, cases, y = cases_y_list(df)
    del df

    start_memory = h.heap().size
    start_time = time.time()

    # generate model
    model = create_model(cases, 100, 3, 1)

    # calculating the average feature vector for each sentence (trace)
    vectors = average_feature_vector(cases, model)

    end_time = time.time() - start_time
    end_memory = h.heap().size - start_memory

    # saving
    out_df = pd.DataFrame(vectors, columns=[f'feature_{i}' for i in range(100)])
    out_df['case'] = ids
    out_df['time'] = end_time
    out_df['memory'] = end_memory
    out_df['label'] = y
    out_df.to_csv(f'{save_path}/{file}', index=False)

    del out_df, model, ids, vectors, y

    print(file, end_time, end_memory)
print('final time', time.time() - st)
