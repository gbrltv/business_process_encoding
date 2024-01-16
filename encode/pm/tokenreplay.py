import time
import pandas as pd
import pm4py
from .utils import retrieve_traces


def format_output(replayed_traces):
    trace_is_fit, trace_fitness, missing_tokens, consumed_tokens, remaining_tokens, produced_tokens = [], [], [], [], [], []
    for replayed in replayed_traces:
        trace_is_fit.append(replayed['trace_is_fit'])
        trace_fitness.append(float(replayed['trace_fitness']))
        missing_tokens.append(float(replayed['missing_tokens']))
        consumed_tokens.append(float(replayed['consumed_tokens']))
        remaining_tokens.append(float(replayed['remaining_tokens']))
        produced_tokens.append(float(replayed['produced_tokens']))

    return [trace_is_fit, trace_fitness, missing_tokens, consumed_tokens, remaining_tokens, produced_tokens]



def run_tokenreplay(config, log):
    ids, traces = retrieve_traces(log)

    start_time = time.time()

    # generate process model
    net, im, fm = pm4py.discover_petri_net_inductive(
        log,
        activity_key="concept:name",
        case_id_key="case:concept:name",
        timestamp_key="time:timestamp",
    )

    # calculating tokenreplay
    tbr_diagnostics = pm4py.conformance_diagnostics_token_based_replay(
        log,
        net,
        im,
        fm,
        activity_key="concept:name",
        case_id_key="case:concept:name",
        timestamp_key="time:timestamp",
    )

    end_time = time.time() - start_time
    print(f"\nToken replay took {round(end_time, 2)} seconds")

    output = format_output(tbr_diagnostics)

    # saving
    out_df = pd.DataFrame(ids, columns=["case"])
    out_df['trace_is_fit'] = output[0]
    out_df['trace_fitness'] = output[1]
    out_df['missing_tokens'] = output[2]
    out_df['consumed_tokens'] = output[3]
    out_df['remaining_tokens'] = output[4]
    out_df['produced_tokens'] = output[5]

    return out_df
