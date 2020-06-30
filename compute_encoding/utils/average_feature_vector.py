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
    vectors = []
    for trace in traces:
        trace_vector = []
        for token in trace:
            try:
                trace_vector.append(model.wv[token])
            except KeyError:
                pass
        vectors.append(np.array(trace_vector).mean(axis=0))

    return vectors
