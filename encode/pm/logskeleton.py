import time
import pandas as pd
import pm4py
from .utils import retrieve_traces


def format_output(alignments):
    no_dev_total, no_constr_total, dev_fitness, is_fit = [], [], [], []
    for alignment in alignments:
        no_dev_total.append(alignment["no_dev_total"])
        no_constr_total.append(alignment["no_constr_total"])
        dev_fitness.append(alignment["dev_fitness"])
        is_fit.append(alignment["is_fit"])

    return [no_dev_total, no_constr_total, dev_fitness, is_fit]

def run_logskeleton(config, log):
    ids, traces = retrieve_traces(log)

    start_time = time.time()

    log_skeleton = pm4py.discover_log_skeleton(
        log,
        noise_threshold=0.1,
        activity_key="concept:name",
        case_id_key="case:concept:name",
        timestamp_key="time:timestamp",
    )
    conformance_lsk = pm4py.conformance_log_skeleton(
        log,
        log_skeleton,
        activity_key="concept:name",
        case_id_key="case:concept:name",
        timestamp_key="time:timestamp",
    )
    end_time = time.time() - start_time
    print(f"\nLog skeleton took {round(end_time, 2)} seconds")

    output = format_output(conformance_lsk)

    # saving
    out_df = pd.DataFrame(ids, columns=["case"])
    out_df["no_dev_total"] = output[0]
    out_df["no_constr_total"] = output[1]
    out_df["dev_fitness"] = output[2]
    out_df["is_fit"] = output[3]

    return out_df
