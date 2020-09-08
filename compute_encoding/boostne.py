import os
import time
import pandas as pd
from tqdm import tqdm
from skmultiflow.utils import calculate_object_size
from karateclub.node_embedding.neighbourhood import BoostNE
import networkx as nx
from utils import read_log
from utils import sort_alphanumeric
from utils import retrieve_traces
from utils import convert_traces_mapping
from utils import create_graph
from utils import trace_feature_vector_from_nodes
from utils import trace_feature_vector_from_edges
import warnings
warnings.filterwarnings('ignore')


def save_results(vector, dimension, ids, time, memory, y, path):
    out_df = pd.DataFrame(vector, columns=[f'feature_{i}' for i in range(dimension)])
    out_df['case'] = ids
    out_df['time'] = time
    out_df['memory'] = memory
    out_df['label'] = y
    out_df.to_csv(path, index=False)


dimensions = [17, 34, 68, 136, 255]
path = './event_logs'
save_path = './encoding_results/boostne'
for type in ['average', 'max']:
    for dimension in dimensions:
        os.makedirs(f'{save_path}/node/{type}/{dimension}', exist_ok=True)
        for edge_type in ['average', 'hadamard', 'weightedl1', 'weightedl2']:
            os.makedirs(f'{save_path}/edge/{edge_type}/{type}/{dimension}', exist_ok=True)

for file in tqdm(sort_alphanumeric(os.listdir(path))):
    # create graph
    file_name = file.split('.csv')[0]
    graph = create_graph(f'./event_logs_xes/{file_name}.xes')
    mapping = dict(zip(graph.nodes(), [i for i in range(len(graph.nodes()))]))
    graph = nx.relabel_nodes(graph, mapping)

    # read event log, import case id and labels and transform activities names
    ids, traces_raw, y = retrieve_traces(read_log(path, file, ' '))
    traces = convert_traces_mapping(traces_raw, mapping)

    for dimension in dimensions:
        start_time = time.time()
        # generate model
        model = BoostNE(dimensions=int(dimension/17))
        model.fit(graph)
        training_time = time.time() - start_time

        # calculating the average and max feature vector for each trace
        start_time = time.time()
        node_average, node_max = trace_feature_vector_from_nodes(model.get_embedding(), traces, dimension)
        node_time = training_time + (time.time() - start_time)

        # saving
        mem_size = calculate_object_size(node_average) + calculate_object_size(model)
        save_results(node_average, dimension, ids, node_time, mem_size, y, f'{save_path}/node/average/{dimension}/{file}')
        mem_size = calculate_object_size(node_max) + calculate_object_size(model)
        save_results(node_max, dimension, ids, node_time, mem_size, y, f'{save_path}/node/max/{dimension}/{file}')

        start_time = time.time()
        edge_average_average, edge_average_max, edge_hadamard_average, edge_hadamard_max, edge_weightedl1_average, edge_weightedl1_max, edge_weightedl2_average, edge_weightedl2_max = trace_feature_vector_from_edges(model.get_embedding(), traces, dimension)
        edge_time = training_time + (time.time() - start_time)

        mem_size = calculate_object_size(edge_average_average) + calculate_object_size(model)
        save_results(edge_average_average, dimension, ids, edge_time, mem_size, y, f'{save_path}/edge/average/average/{dimension}/{file}')
        mem_size = calculate_object_size(edge_average_max) + calculate_object_size(model)
        save_results(edge_average_max, dimension, ids, edge_time, mem_size, y, f'{save_path}/edge/average/max/{dimension}/{file}')

        mem_size = calculate_object_size(edge_hadamard_average) + calculate_object_size(model)
        save_results(edge_hadamard_average, dimension, ids, edge_time, mem_size, y, f'{save_path}/edge/hadamard/average/{dimension}/{file}')
        mem_size = calculate_object_size(edge_hadamard_max) + calculate_object_size(model)
        save_results(edge_hadamard_max, dimension, ids, edge_time, mem_size, y, f'{save_path}/edge/hadamard/max/{dimension}/{file}')

        mem_size = calculate_object_size(edge_weightedl1_average) + calculate_object_size(model)
        save_results(edge_weightedl1_average, dimension, ids, edge_time, mem_size, y, f'{save_path}/edge/weightedl1/average/{dimension}/{file}')
        mem_size = calculate_object_size(edge_weightedl1_max) + calculate_object_size(model)
        save_results(edge_weightedl1_max, dimension, ids, edge_time, mem_size, y, f'{save_path}/edge/weightedl1/max/{dimension}/{file}')

        mem_size = calculate_object_size(edge_weightedl2_average) + calculate_object_size(model)
        save_results(edge_weightedl2_average, dimension, ids, edge_time, mem_size, y, f'{save_path}/edge/weightedl2/average/{dimension}/{file}')
        mem_size = calculate_object_size(edge_weightedl2_max) + calculate_object_size(model)
        save_results(edge_weightedl2_max, dimension, ids, edge_time, mem_size, y, f'{save_path}/edge/weightedl2/max/{dimension}/{file}')
