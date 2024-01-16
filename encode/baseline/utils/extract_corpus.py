def extract_corpus(log):
    traces, ids = [], []
    for id in log["case:concept:name"].unique():
        events = list(log[log["case:concept:name"] == id]["concept:name"])
        traces.append(" ".join(x.replace(" ", "") for x in events))
        ids.append(id)

    return ids, traces
