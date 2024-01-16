import pm4py

def read_log(path: str):
    log = pm4py.read_xes(f"{path}")
    log["case:concept:name"] = log["case:concept:name"].astype("string")
    log["concept:name"] = log["concept:name"].astype("string")

    return log
