import re
import os
import time
import pandas as pd
from guppy import hpy; h=hpy()
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.conformance.alignments import algorithm as alignments
from pm4py.algo.conformance.alignments import versions

def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
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
save_path = 'encoding_results/alignment'
os.makedirs(save_path, exist_ok=True)
st = time.time()
for file in sorted_aphanumeric(os.listdir(path))[261:]:
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

    # compute alignments
    trace_alignments = alignments.apply_log(log, net, initial_marking, final_marking, variant=versions.dijkstra_no_heuristics)

    end_time = time.time() - start_time
    end_memory = h.heap().size - start_memory


    cost, visited_states, queued_states, traversed_arcs, fitness = [], [], [], [], []
    for alignment in trace_alignments:
        cost.append(alignment['cost'])
        visited_states.append(alignment['visited_states'])
        queued_states.append(alignment['queued_states'])
        traversed_arcs.append(alignment['traversed_arcs'])
        fitness.append(alignment['fitness'])

    # saving
    out_df = pd.DataFrame()
    out_df['cost'] = cost
    out_df['visited_states'] = visited_states
    out_df['queued_states'] = queued_states
    out_df['traversed_arcs'] = traversed_arcs
    out_df['fitness'] = fitness
    out_df['case'] = ids
    out_df['time'] = end_time
    out_df['memory'] = end_memory
    out_df['label'] = y
    out_df.to_csv(f'{save_path}/{file}', index=False)

    del log, trace_alignments, out_df, cost, visited_states, queued_states, traversed_arcs, fitness, net, initial_marking, final_marking, ids, y

    print(file, end_time, end_memory)
print('final time', time.time() - st)
