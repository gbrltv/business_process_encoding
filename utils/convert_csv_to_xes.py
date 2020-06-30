import os
import pandas as pd
from tqdm import tqdm
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.util import constants
from sort_alphanumeric import sort_alphanumeric

path = './event_logs'
out_path = './event_logs_xes'
os.makedirs(out_path, exist_ok=True)
for file in tqdm(sort_alphanumeric(os.listdir(path))):
    df = pd.read_csv(f'{path}/{file}')

    df.columns = ['case:concept:name', 'concept:name', 'timestamp',
        'actor', 'resource', 'label', 'description']

    df = dataframe_utils.convert_timestamp_columns_in_df(df)
    params = {
        constants.PARAMETER_CONSTANT_CASEID_KEY: 'case:concept:name',
        constants.PARAMETER_CONSTANT_ACTIVITY_KEY: 'concept:name',
        constants.PARAMETER_CONSTANT_TIMESTAMP_KEY: 'timestamp'}
    event_log = log_converter.apply(df, parameters=params, variant=log_converter.Variants.TO_EVENT_LOG)

    out_file_name = file.split('.csv')[0] + '.xes'
    xes_exporter.apply(event_log, f'{out_path}/{out_file_name}')
