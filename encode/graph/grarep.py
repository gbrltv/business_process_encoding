import time
import pandas as pd
from karateclub.node_embedding.neighbourhood import GraRep
import networkx as nx
from .utils import (
    convert_traces_mapping,
    retrieve_traces,
    create_graph,
    trace_feature_vector_from_nodes,
    trace_feature_vector_from_edges,
)
import warnings

warnings.filterwarnings("ignore")


def run_grarep(config, log):
    ids, traces_raw = retrieve_traces(log)

    start_time = time.time()

    # create graph
    graph = create_graph(log)
    mapping = dict(zip(graph.nodes(), [i for i in range(len(graph.nodes()))]))
    graph = nx.relabel_nodes(graph, mapping)
    traces = convert_traces_mapping(traces_raw, mapping)

    # generate model
    model = GraRep(dimensions=config["vector_size"])
    model.fit(graph)

    # calculating feature vectors for each trace
    if config["embed_from"] == "nodes":
        vectors = trace_feature_vector_from_nodes(model.get_embedding(), traces, config["vector_size"], aggregation=config["aggregation"])
    elif config["embed_from"] == "edges":
        vectors = trace_feature_vector_from_edges(model.get_embedding(), traces, config["vector_size"], aggregation=config["aggregation"], edge_operator=config["edge_operator"])
    else:
        raise Exception("Please choose a valid graph embedding origin: {nodes, edges}")

    end_time = time.time() - start_time
    print(f"\nGraRep took {round(end_time, 2)} seconds")

    encoded_df = pd.DataFrame(vectors, columns=[f'{i}' for i in range(len(vectors[0]))])
    encoded_df.insert(0, "case", ids)

    return encoded_df
