from node2vec import Node2Vec
from node2vec.edges import HadamardEmbedder


def train_text_model(model, cases):
    """
    Creates a text model

    Parameters
    -----------------------
    model,
        Text-based model containing the computed encodings
    cases: List,
        List of cases treated as sentences by the model
    Returns
    -----------------------
    model:
        Trained text-based model containing the computed encodings
    """
    model.build_vocab(cases)
    model.train(cases, total_examples=len(cases), epochs=10)

    return model


def train_graph_model(graph, dimensions: int = 128, window: int = 10, min_count: int = 1, is_edges: bool = False):
    """
    Creates a node2vec model

    Parameters
    -----------------------
    graph: node2vec,
        Graph-based model containing the computed encodings
    dimensions: int,
        Number of encoding dimensions
    window: int,
        Encoding window size
    min_count: int,
        Minimum number of appearences for a given word
    Returns
    -----------------------
    model:
        Trained graph-based model containing the computed encodings
    """
    model = Node2Vec(graph, dimensions=dimensions, num_walks=50)
    model = model.fit(window=window, min_count=min_count, workers=-1)

    if is_edges:
        model = HadamardEmbedder(keyed_vectors=model.wv)
    return model
