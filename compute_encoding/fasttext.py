import os
import time
import pandas as pd
from tqdm import tqdm
from gensim.models import FastText
from skmultiflow.utils import calculate_object_size
from utils import read_log
from utils import sort_alphanumeric
from utils import retrieve_traces
from utils import train_text_model
from utils import average_feature_vector


def save_results(vector, dimension, ids, time, memory, y, path):
    out_df = pd.DataFrame(vector, columns=[f'feature_{i}' for i in range(dimension)])
    out_df['case'] = ids
    out_df['time'] = time
    out_df['memory'] = memory
    out_df['label'] = y
    out_df.to_csv(path, index=False)


dimensions = [2, 4, 8, 16, 32, 64, 128, 256]
path = './event_logs'
save_path = './encoding_results/fasttext'
for type in ['average', 'max']:
    for dimension in dimensions:
        os.makedirs(f'{save_path}/{type}/{dimension}', exist_ok=True)

for file in tqdm(sort_alphanumeric(os.listdir(path))):
    # read event log and import case id and labels
    ids, traces, y = retrieve_traces(read_log(path, file))

    for dimension in dimensions:
        start_time = time.time()
        # generate model
        model = FastText(size=dimension, window=3, min_count=1, workers=-1)
        model = train_text_model(model, traces)

        # calculating the average feature vector for each sentence (trace)
        vectors_average, vectors_max = average_feature_vector(model, traces)

        end_time = time.time() - start_time

        mem_size = calculate_object_size(vectors_average) + calculate_object_size(model)
        save_results(vectors_average, dimension, ids, end_time, mem_size, y, f'{save_path}/average/{dimension}/{file}')
        mem_size = calculate_object_size(vectors_max) + calculate_object_size(model)
        save_results(vectors_max, dimension, ids, end_time, mem_size, y, f'{save_path}/max/{dimension}/{file}')
