import pandas as pd

def extract_corpus(df: pd.DataFrame):
    """
    Creates corpus for model training

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
        traces.append(' '.join(x for x in events))
        y.append(list(group[1].label)[0])
        ids.append(list(group[1].case)[0])

    return ids, traces, y
