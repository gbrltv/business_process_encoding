def convert_traces_mapping(traces_raw, mapping):
    traces = []
    for trace in traces_raw:
        current_trace = []
        for act in trace:
            current_trace.append(mapping[act])
        traces.append(current_trace)
    return traces

def retrieve_traces(log):
    traces, ids = [], []
    for id in log["case:concept:name"].unique():
        events = list(log[log["case:concept:name"] == id]["concept:name"])
        traces.append(["".join(x) for x in events])
        ids.append(id)

    return ids, traces
