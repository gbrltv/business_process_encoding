import os
import time
import pandas as pd
from tqdm import tqdm
from glove import Glove
from glove import Corpus
from skmultiflow.utils import calculate_object_size
from utils import read_log
from utils import sort_alphanumeric
from utils import retrieve_traces
from utils import average_feature_vector_glove


def prepare_traces(traces):
    for trace in traces:
        yield trace


def save_results(vector, dimension, ids, time, memory, y, path):
    out_df = pd.DataFrame(vector, columns=[f'feature_{i}' for i in range(dimension)])
    out_df['case'] = ids
    out_df['time'] = time
    out_df['memory'] = memory
    out_df['label'] = y
    out_df.to_csv(path, index=False)


dimensions = [2, 4, 8, 16, 32, 64, 128, 256]
path = './event_logs'
save_path = './encoding_results/glove'
for type in ['average', 'max']:
    for dimension in dimensions:
        os.makedirs(f'{save_path}/{type}/{dimension}', exist_ok=True)

for file in tqdm(sort_alphanumeric(os.listdir(path))):
    # read event log and import case id and labels
    ids, traces_, y = retrieve_traces(read_log(path, file))

    for dimension in dimensions:
        start_time = time.time()
        # generate model
        model = Corpus()
        model.fit(prepare_traces(traces_))

        glove = Glove(no_components=dimension)
        glove.fit(model.matrix, epochs=10, no_threads=8)
        glove.add_dictionary(model.dictionary)

        # calculating the average feature vector for each sentence (trace)
        vectors_average, vectors_max = average_feature_vector_glove(glove, traces_)

        end_time = time.time() - start_time

        mem_size = calculate_object_size(vectors_average) + calculate_object_size(model) + calculate_object_size(glove)
        save_results(vectors_average, dimension, ids, end_time, mem_size, y, f'{save_path}/average/{dimension}/{file}')
        mem_size = calculate_object_size(vectors_max) + calculate_object_size(model) + calculate_object_size(glove)
        save_results(vectors_max, dimension, ids, end_time, mem_size, y, f'{save_path}/max/{dimension}/{file}')
