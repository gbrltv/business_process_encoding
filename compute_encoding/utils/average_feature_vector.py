import numpy as np


def average_feature_vector(model, traces):
    """
    Computes average feature vector for each trace

    Parameters
    -----------------------
    model,
        Text-based model containing the computed encodings
    traces: List,
        List of traces treated as sentences by the model
    Returns
    -----------------------
    vectors: List
        list of vector encodings for each trace
    """
    vectors_average, vectors_max = [], []
    for trace in traces:
        trace_vector = []
        for token in trace:
            try:
                trace_vector.append(model.wv[token])
            except KeyError:
                pass
        vectors_average.append(np.array(trace_vector).mean(axis=0))
        vectors_max.append(np.array(trace_vector).max(axis=0))

    return vectors_average, vectors_max


def average_feature_vector_doc2vec(model, traces):
    """
    Retrieves the document feature vector for doc2vec

    Parameters
    -----------------------
    model,
        Text-based model containing the computed encodings
    traces: List,
        List of traces treated as sentences by the model
    Returns
    -----------------------
    vectors: List
        list of vector encodings for each trace
    """
    vectors = []
    for trace in traces:
        vectors.append(model.infer_vector(trace))

    return vectors

def average_feature_vector_glove(model, traces):
    """
    Retrieves the document feature vector for glove

    Parameters
    -----------------------
    model,
        Text-based model containing the computed encodings
    traces: List,
        List of traces treated as sentences by the model
    Returns
    -----------------------
    vectors: List
        list of vector encodings for each trace
    """
    vectors_average, vectors_max = [], []
    for trace in traces:
        trace_vector = []
        for token in trace:
            try:
                trace_vector.append(model.word_vectors[model.dictionary[token]])
            except KeyError:
                pass
        # print(trace)
        # print(trace_vector)
        vectors_average.append(np.array(trace_vector).mean(axis=0))
        vectors_max.append(np.array(trace_vector).max(axis=0))

    return vectors_average, vectors_max


def trace_feature_vector_from_nodes(embeddings, traces, dimension):
    """
    Computes average feature vector for each trace

    Parameters
    -----------------------
    embeddings,
        Text-based model containing the computed encodings
    traces: List,
        List of traces treated as sentences by the model
    Returns
    -----------------------
    vectors: List
        list of vector encodings for each trace
    """
    vectors_average, vectors_max = [], []
    for trace in traces:
        trace_vector = []
        for token in trace:
            try:
                trace_vector.append(embeddings[token])
            except KeyError:
                pass
        if len(trace_vector) == 0:
            trace_vector.append(np.zeros(dimension))
        vectors_average.append(np.array(trace_vector).mean(axis=0))
        vectors_max.append(np.array(trace_vector).max(axis=0))

    return vectors_average, vectors_max


def trace_feature_vector_from_edges(embeddings, traces, dimension):
    """
    Computes average feature vector for each trace

    Parameters
    -----------------------
    embeddings,
        Text-based model containing the computed encodings
    traces: List,
        List of traces treated as sentences by the model
    Returns
    -----------------------
    vectors: List
        list of vector encodings for each trace
    """
    vectors_average_average, vectors_average_max = [], []
    vectors_hadamard_average, vectors_hadamard_max = [], []
    vectors_weightedl1_average, vectors_weightedl1_max = [], []
    vectors_weightedl2_average, vectors_weightedl2_max = [], []

    for trace in traces:
        trace_vector_average, trace_vector_hadamard, trace_vector_weightedl1, trace_vector_weightedl2 = [], [], [], []
        for i in range(len(trace)-1):
            try:
                emb1, emb2 = embeddings[trace[i]], embeddings[trace[i+1]]
                trace_vector_average.append((emb1 + emb2)/2.0)
                trace_vector_hadamard.append(np.multiply(emb1, emb2))
                trace_vector_weightedl1.append(np.abs(emb1 - emb2))
                trace_vector_weightedl2.append(np.power(np.abs(emb1 - emb2), 2))
            except KeyError:
                pass
        if len(trace_vector_average) == 0:
            trace_vector_average.append(np.zeros(dimension))
            trace_vector_hadamard.append(np.zeros(dimension))
            trace_vector_weightedl1.append(np.zeros(dimension))
            trace_vector_weightedl2.append(np.zeros(dimension))

        vectors_average_average.append(np.array(trace_vector_average).mean(axis=0))
        vectors_average_max.append(np.array(trace_vector_average).max(axis=0))

        vectors_hadamard_average.append(np.array(trace_vector_hadamard).mean(axis=0))
        vectors_hadamard_max.append(np.array(trace_vector_hadamard).max(axis=0))

        vectors_weightedl1_average.append(np.array(trace_vector_weightedl1).mean(axis=0))
        vectors_weightedl1_max.append(np.array(trace_vector_weightedl1).max(axis=0))

        vectors_weightedl2_average.append(np.array(trace_vector_weightedl2).mean(axis=0))
        vectors_weightedl2_max.append(np.array(trace_vector_weightedl2).max(axis=0))

    return vectors_average_average, vectors_average_max, vectors_hadamard_average, vectors_hadamard_max, vectors_weightedl1_average, vectors_weightedl1_max, vectors_weightedl2_average, vectors_weightedl2_max
