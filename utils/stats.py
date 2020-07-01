import glob
import pandas as pd
import numpy as np


scenarios = ['1', '2', '3', '4', '5']
for complexity in scenarios:
    n_cases, n_activities, n_events, n_attr_values = [], [], [], []
    for log in glob.glob(f'event_logs/scenario{complexity}*_1000_*'):
        df = pd.read_csv(log)

        n_events.append(len(df))
        n_cases.append(len(set(df['case'])))

        trace_len = []
        for case in set(df['case']):
            trace_len.append(len(df[df['case'] == case]))

        acts = list(set(df['activity']))
        n_activities.append(len([x for x in acts if 'Random' not in x])) # filtering anomalous instances

        actors = list(set(df['actor']))
        n_actors = len([x for x in actors if 'Random' not in x]) # filtering anomalous instances
        resources = list(set(df['resource']))
        n_resources = len([x for x in resources if x%10==0]) # filtering anomalous instances
        n_attr_values.append(n_actors+n_resources)

    print(f'scenario{complexity}')
    print('#logs', len(n_cases))
    print('#events' , np.min(n_events), '-', np.max(n_events))
    print('#cases' , np.min(n_cases), '-', np.max(n_cases))
    print('#trace length' , np.min(trace_len), '-', np.max(trace_len))
    print('#activities' , np.min(n_activities), '-', np.max(n_activities))
    print('#attribute_values' , np.min(n_attr_values), '-', np.max(n_attr_values))
    print()
