import numpy as np

def trace_feature_vector_from_nodes(embeddings, traces, dimension, aggregation="average"):
    vectors = []
    for trace in traces:
        trace_vector = []
        for token in trace:
            try:
                trace_vector.append(embeddings[token])
            except KeyError:
                pass
        if len(trace_vector) == 0:
            trace_vector.append(np.zeros(dimension))
        if aggregation == "average":
            vectors.append(np.array(trace_vector).mean(axis=0))
        elif aggregation == "max":
            vectors.append(np.array(trace_vector).max(axis=0))
        else:
            raise Exception("Please select a valid aggregation method: {average, max}")

    return vectors


def trace_feature_vector_from_edges(embeddings, traces, dimension, aggregation="average", edge_operator="average"):
    vectors = []

    for trace in traces:
        trace_vector = []
        for i in range(len(trace)-1):
            try:
                emb1, emb2 = embeddings[trace[i]], embeddings[trace[i+1]]
                if edge_operator == "average":
                    trace_vector.append((emb1 + emb2)/2.0)
                elif edge_operator == "hadamard":
                    trace_vector.append(np.multiply(emb1, emb2))
                elif edge_operator == "weightedl1":
                    trace_vector.append(np.abs(emb1 - emb2))
                elif edge_operator == "weightedl2":
                    trace_vector.append(np.power(np.abs(emb1 - emb2), 2))
                else:
                    raise Exception("Please select a valid edge operator: {average, hadamard, weightedl1, weightedl2}")
            except KeyError:
                pass
        if len(trace_vector) == 0:
            trace_vector.append(np.zeros(dimension))

        if aggregation == "average":
            vectors.append(np.array(trace_vector).mean(axis=0))
        elif aggregation == "max":
            vectors.append(np.array(trace_vector).max(axis=0))

    return vectors
