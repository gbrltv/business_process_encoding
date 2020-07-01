import os
import time
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from utils import read_log
from utils import sort_alphanumeric
from utils import extract_corpus


path = './event_logs'
save_path = './encoding_results/countvectorizer'
os.makedirs(save_path, exist_ok=True)
for file in tqdm(sort_alphanumeric(os.listdir(path))):
    # read event log and import case id and labels
    ids, traces, y = extract_corpus(read_log(path, file, ''))

    start_time = time.time()

    # generate model
    model = CountVectorizer()
    encoding = model.fit_transform(traces)

    end_time = time.time() - start_time

    # saving
    out_df = pd.DataFrame(encoding.toarray(), columns=[f'feature_{i}' for i in range(encoding.toarray().shape[1])])
    out_df['case'] = ids
    out_df['time'] = end_time
    out_df['label'] = y
    out_df.to_csv(f'{save_path}/{file}', index=False)
