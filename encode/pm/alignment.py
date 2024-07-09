import time
import pandas as pd
import pm4py
from .utils import retrieve_traces


def format_output(alignments):
    cost, visited_states, queued_states, traversed_arcs, lp_solved, fitness, bwc = [], [], [], [], [], [], []
    for alignment in alignments:
        cost.append(alignment["cost"])
        visited_states.append(alignment["visited_states"])
        queued_states.append(alignment["queued_states"])
        traversed_arcs.append(alignment["traversed_arcs"])
        lp_solved.append(alignment["lp_solved"])
        fitness.append(alignment["fitness"])
        bwc.append(alignment["bwc"])

    return [cost, visited_states, queued_states, traversed_arcs, lp_solved, fitness, bwc]



def run_alignment(config, log):
    ids, traces = retrieve_traces(log)

    start_time = time.time()

    # generate process model
    net, im, fm = pm4py.discover_petri_net_inductive(
        log,
        activity_key="concept:name",
        case_id_key="case:concept:name",
        timestamp_key="time:timestamp",
    )

    # compute alignments
    alignments_diagnostics = pm4py.conformance_diagnostics_alignments(
        log,
        net,
        im,
        fm,
        activity_key="concept:name",
        case_id_key="case:concept:name",
        timestamp_key="time:timestamp",
        # multi_processing=True,
    )

    end_time = time.time() - start_time
    print(f"\nAlignments took {round(end_time, 2)} seconds")

    output = format_output(alignments_diagnostics)

    # saving
    out_df = pd.DataFrame(ids, columns=["case"])
    out_df["cost"] = output[0]
    out_df["visited_states"] = output[1]
    out_df["queued_states"] = output[2]
    out_df["traversed_arcs"] = output[3]
    out_df["lp_solved"] = output[4]
    out_df["fitness"] = output[5]
    out_df["bwc"] = output[6]

    return out_df
