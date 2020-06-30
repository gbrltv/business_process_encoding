import re
import os
import time
import pandas as pd
from guppy import hpy; h=hpy()
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay

def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)

def read_log(path, log):
    """
    Reads event log and preprocess it
    """
    df_raw = pd.read_csv(f'{path}/{log}')
    df_raw['activity_name'] = df_raw['activity'].str.replace(' ', '-')
    df_proc = df_raw[['case', 'activity_name', 'label']]
    del df_raw
    return df_proc

def cases_y_list(df):
    """
    Creates a list of cases for model training
    """
    cases, y, ids = [], [], []
    for group in df.groupby('case'):
        events = list(group[1].activity_name)
        cases.append([''.join(x) for x in events])
        y.append(list(group[1].label)[0])
        ids.append(list(group[1].case)[0])

    return ids, cases, y


path = 'event_logs'
save_path = 'encoding_results/tokenreplay'
os.makedirs(save_path, exist_ok=True)
st = time.time()
for file in sorted_aphanumeric(os.listdir(path))[232:]:
    # read event log and import case id and labels
    ids, cases, y = cases_y_list(read_log(path, file))
    del cases

    # import log
    file_xes = file.split('.csv')[0]
    log = xes_importer.apply(f'{path}_xes/{file_xes}.xes')

    start_memory = h.heap().size
    start_time = time.time()

    # generate process model
    net, initial_marking, final_marking = inductive_miner.apply(log)

    # calculating tokenreplay
    replayed_traces = token_replay.apply(log, net, initial_marking, final_marking)

    end_time = time.time() - start_time
    end_memory = h.heap().size - start_memory

    trace_is_fit, trace_fitness, missing_tokens, consumed_tokens, remaining_tokens, produced_tokens = [], [], [], [], [], []
    for replayed in replayed_traces:
        trace_is_fit.append(replayed['trace_is_fit'])
        trace_fitness.append(float(replayed['trace_fitness']))
        missing_tokens.append(float(replayed['missing_tokens']))
        consumed_tokens.append(float(replayed['consumed_tokens']))
        remaining_tokens.append(float(replayed['remaining_tokens']))
        produced_tokens.append(float(replayed['produced_tokens']))

    # saving
    out_df = pd.DataFrame()
    out_df['trace_is_fit'] = trace_is_fit
    out_df['trace_fitness'] = trace_fitness
    out_df['missing_tokens'] = missing_tokens
    out_df['consumed_tokens'] = consumed_tokens
    out_df['remaining_tokens'] = remaining_tokens
    out_df['produced_tokens'] = produced_tokens
    out_df['case'] = ids
    out_df['time'] = end_time
    out_df['memory'] = end_memory
    out_df['label'] = y
    out_df.to_csv(f'{save_path}/{file}', index=False)

    del log, net, initial_marking, final_marking, replayed_traces, trace_is_fit, trace_fitness, missing_tokens, consumed_tokens, remaining_tokens, produced_tokens, out_df

    print(file, end_time, end_memory)
print('final time', time.time() - st)
