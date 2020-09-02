import os
import time
import pandas as pd
from tqdm import tqdm
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.conformance.alignments import algorithm as alignments
from pm4py.algo.conformance.alignments import versions
from skmultiflow.utils import calculate_object_size
from utils import read_log
from utils import sort_alphanumeric
from utils import retrieve_traces


def compute_alignments(alignments):
    cost, visited_states, queued_states, traversed_arcs, fitness = [], [], [], [], []
    for alignment in trace_alignments:
        if alignment is None:
            cost.append(0)
            visited_states.append(0)
            queued_states.append(0)
            traversed_arcs.append(0)
            fitness.append(0)
        else:
            cost.append(alignment['cost'])
            visited_states.append(alignment['visited_states'])
            queued_states.append(alignment['queued_states'])
            traversed_arcs.append(alignment['traversed_arcs'])
            fitness.append(alignment['fitness'])

    return [cost, visited_states, queued_states, traversed_arcs, fitness]


path = './event_logs'
save_path = './encoding_results/alignment'
os.makedirs(save_path, exist_ok=True)
parameters = {}
parameters[alignments.Parameters.PARAM_MAX_ALIGN_TIME_TRACE] = 1
for file in tqdm(sort_alphanumeric(os.listdir(path))):
    # read event log and import case id and labels
    ids, traces, y = retrieve_traces(read_log(path, file))

    # import log
    file_xes = file.split('.csv')[0]
    log = xes_importer.apply(f'{path}_xes/{file_xes}.xes')

    start_time = time.time()

    # generate process model
    net, initial_marking, final_marking = inductive_miner.apply(log)

    # compute alignments
    trace_alignments = alignments.apply_log(log, net, initial_marking, final_marking, parameters=parameters, variant=versions.dijkstra_no_heuristics)

    end_time = time.time() - start_time
    memory = calculate_object_size(trace_alignments)

    final_alignments = compute_alignments(trace_alignments)

    # saving
    out_df = pd.DataFrame()
    out_df['cost'] = final_alignments[0]
    out_df['visited_states'] = final_alignments[1]
    out_df['queued_states'] = final_alignments[2]
    out_df['traversed_arcs'] = final_alignments[3]
    out_df['fitness'] = final_alignments[4]
    out_df['case'] = ids
    out_df['time'] = end_time
    out_df['memory'] = memory
    out_df['label'] = y
    out_df.to_csv(f'{save_path}/{file}', index=False)
