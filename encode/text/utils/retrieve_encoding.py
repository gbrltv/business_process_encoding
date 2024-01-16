import numpy as np

def retrieve_encoding(model, traces, aggregation="average"):
    vectors = []
    for trace in traces:
        trace_vector = []
        for token in trace:
            try:
                trace_vector.append(model.wv[token])
            except KeyError:
                pass
        if aggregation == "average":
            vectors.append(np.array(trace_vector).mean(axis=0))
        elif aggregation == "max":
            vectors.append(np.array(trace_vector).max(axis=0))
        else:
            raise Exception(
                "Please select a valid aggregation method: {average, max}"
            )
    return vectors
