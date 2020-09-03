import os
import time
import pandas as pd
from tqdm import tqdm
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.log_skeleton import algorithm as lsk_discovery
from pm4py.algo.conformance.log_skeleton import algorithm as lsk_conformance
from skmultiflow.utils import calculate_object_size
from utils import read_log
from utils import sort_alphanumeric
from utils import retrieve_traces


def compute_alignments(alignments):
    no_dev_total, no_constr_total, dev_fitness, is_fit = [], [], [], []
    for alignment in alignments:
        no_dev_total.append(alignment['no_dev_total'])
        no_constr_total.append(alignment['no_constr_total'])
        dev_fitness.append(alignment['dev_fitness'])
        is_fit.append(alignment['is_fit'])

    return [no_dev_total, no_constr_total, dev_fitness, is_fit]


path = './event_logs'
save_path = './encoding_results/log_skeleton'
os.makedirs(save_path, exist_ok=True)
for file in tqdm(sort_alphanumeric(os.listdir(path))):
    # read event log and import case id and labels
    ids, traces, y = retrieve_traces(read_log(path, file))

    # import log
    file_xes = file.split('.csv')[0]
    log = xes_importer.apply(f'{path}_xes/{file_xes}.xes')

    start_time = time.time()

    skeleton = lsk_discovery.apply(log, parameters={lsk_discovery.Variants.CLASSIC.value.Parameters.NOISE_THRESHOLD: 0.05})
    conf_result = lsk_conformance.apply(log, skeleton)

    end_time = time.time() - start_time
    memory = calculate_object_size(conf_result) + calculate_object_size(skeleton)

    final_alignments = compute_alignments(conf_result)

    # saving
    out_df = pd.DataFrame()
    out_df['no_dev_total'] = final_alignments[0]
    out_df['no_constr_total'] = final_alignments[1]
    out_df['dev_fitness'] = final_alignments[2]
    out_df['is_fit'] = final_alignments[3]
    out_df['case'] = ids
    out_df['time'] = end_time
    out_df['memory'] = memory
    out_df['label'] = y
    out_df.to_csv(f'{save_path}/{file}', index=False)
