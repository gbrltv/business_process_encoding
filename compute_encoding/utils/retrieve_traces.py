import pandas as pd


def retrieve_traces(df: pd.DataFrame):
    """
    Creates a list of cases for model training

    Parameters
    -----------------------
    df: pd.DataFrame,
        Dataframe containing the event log
    Returns
    -----------------------
    ids: List,
        List of case ids
    traces: List,
        List of traces
    y: List,
        List of case labels
    """
    traces, y, ids = [], [], []
    for group in df.groupby('case'):
        events = list(group[1].activity_name)
        traces.append([''.join(x) for x in events])
        y.append(list(group[1].label)[0])
        ids.append(list(group[1].case)[0])

    return ids, traces, y

def convert_traces_mapping(traces_raw, mapping):
    """
    Convert traces activity name using a given mapping

    Parameters
    -----------------------
    traces: List,
        List of traces
    mapping: dict:
        Dictionary containing activities mapping
    Returns
    -----------------------
        List of converted traces
    """
    traces = []
    for trace in traces_raw:
        current_trace = []
        for act in trace:
            current_trace.append(mapping[act])
        traces.append(current_trace)
    return traces
