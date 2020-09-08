import os
import time
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import HashingVectorizer
from skmultiflow.utils import calculate_object_size
from utils import read_log
from utils import sort_alphanumeric
from utils import extract_corpus


dimensions = [2, 4, 8, 16, 32, 64, 128, 256]
path = './event_logs'
save_path = './encoding_results/hashvectorizer'
for dimension in dimensions:
    os.makedirs(f'{save_path}/{dimension}', exist_ok=True)

for file in tqdm(sort_alphanumeric(os.listdir(path))):
    # read event log and import case id and labels
    ids, traces, y = extract_corpus(read_log(path, file, ''))

    for dimension in dimensions:
        start_time = time.time()
        # generate model
        model = HashingVectorizer(n_features=dimension)
        encoding = model.fit_transform(traces)

        end_time = time.time() - start_time
        mem_size = calculate_object_size(encoding) + calculate_object_size(model)

        # saving
        out_df = pd.DataFrame(encoding.toarray(), columns=[f'feature_{i}' for i in range(encoding.toarray().shape[1])])
        out_df['case'] = ids
        out_df['time'] = end_time
        out_df['memory'] = mem_size
        out_df['label'] = y
        out_df.to_csv(f'{save_path}/{dimension}/{file}', index=False)
