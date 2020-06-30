def extract_corpus(df):
    """
    Creates corpus for model training
    """
    cases, y, ids = [], [], []
    for group in df.groupby('case'):
        events = list(group[1].activity_name)
        cases.append(' '.join(x for x in events))
        y.append(list(group[1].label)[0])
        ids.append(list(group[1].case)[0])

    return ids, cases, y
