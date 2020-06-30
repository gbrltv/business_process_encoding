import pandas as pd


def read_log(path: str, log: str, replace_space: str = '-') -> pd.DataFrame:
    """
    Reads event log and preprocess it

    Parameters
    -----------------------
    path: str,
        File path
    log: str,
        File name
    replace_space: str,
        Replace space from activity name
    Returns
    -----------------------
    Processed event log containing the only the necessary columns for encoding
    """
    df_raw = pd.read_csv(f'{path}/{log}')
    df_raw['activity_name'] = df_raw['activity'].str.replace(' ', replace_space)
    df_proc = df_raw[['case', 'activity_name', 'label']]

    return df_proc
