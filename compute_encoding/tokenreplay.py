import os
import time
import pandas as pd
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from utils import read_log
from utils import sort_alphanumeric
from utils import retrieve_traces


def compute_alignments(alignments):
    trace_is_fit, trace_fitness, missing_tokens, consumed_tokens, remaining_tokens, produced_tokens = [], [], [], [], [], []
    for replayed in replayed_traces:
        trace_is_fit.append(replayed['trace_is_fit'])
        trace_fitness.append(float(replayed['trace_fitness']))
        missing_tokens.append(float(replayed['missing_tokens']))
        consumed_tokens.append(float(replayed['consumed_tokens']))
        remaining_tokens.append(float(replayed['remaining_tokens']))
        produced_tokens.append(float(replayed['produced_tokens']))

    return [trace_is_fit, trace_fitness, missing_tokens, consumed_tokens, remaining_tokens, produced_tokens]


path = './event_logs'
save_path = './encoding_results/tokenreplay'
os.makedirs(save_path, exist_ok=True)
for file in sort_alphanumeric(os.listdir(path)):
    # read event log and import case id and labels
    ids, traces, y = retrieve_traces(read_log(path, file))

    # import xes log for process discovery
    file_xes = file.split('.csv')[0]
    log = xes_importer.apply(f'{path}_xes/{file_xes}.xes')

    start_time = time.time()

    # generate process model
    net, initial_marking, final_marking = inductive_miner.apply(log)

    # calculating tokenreplay
    replayed_traces = token_replay.apply(log, net, initial_marking, final_marking)

    end_time = time.time() - start_time

    final_token_replay = compute_alignments(replayed_traces)


    # saving
    out_df = pd.DataFrame()
    out_df['trace_is_fit'] = final_token_replay[0]
    out_df['trace_fitness'] = final_token_replay[1]
    out_df['missing_tokens'] = final_token_replay[2]
    out_df['consumed_tokens'] = final_token_replay[3]
    out_df['remaining_tokens'] = final_token_replay[4]
    out_df['produced_tokens'] = final_token_replay[5]
    out_df['case'] = ids
    out_df['time'] = end_time
    out_df['label'] = y
    out_df.to_csv(f'{save_path}/{file}', index=False)
