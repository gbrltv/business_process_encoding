import os
import time
import pandas as pd
from tqdm import tqdm
from skmultiflow.utils import calculate_object_size
from utils import read_log
from utils import sort_alphanumeric
from utils import retrieve_traces
from utils import average_feature_vector
from utils import create_graph
from utils import train_graph_model


path = './event_logs'
save_path = './encoding_results/node2vec'
os.makedirs(save_path, exist_ok=True)
for file in tqdm(sort_alphanumeric(os.listdir(path))):
    # read event log and import case id and labels
    ids, traces, y = retrieve_traces(read_log(path, file, ' '))

    start_time = time.time()

    # create graph
    file_name = file.split('.csv')[0]
    graph = create_graph(f'./event_logs_xes/{file_name}.xes')
    # generate model
    model = train_graph_model(graph)

    # calculating the average feature vector for each sentence (trace)
    vectors = average_feature_vector(model, traces)

    end_time = time.time() - start_time
    memory = calculate_object_size(vectors)

    # saving
    out_df = pd.DataFrame(vectors, columns=[f'feature_{i}' for i in range(128)])
    out_df['case'] = ids
    out_df['time'] = end_time
    out_df['memory'] = memory
    out_df['label'] = y
    out_df.to_csv(f'{save_path}/{file}', index=False)
