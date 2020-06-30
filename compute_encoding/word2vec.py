import os
import time
import pandas as pd
from tqdm import tqdm
from gensim.models import Word2Vec
from utils import read_log
from utils import sort_alphanumeric
from utils import retrieve_traces
from utils import train_text_model
from utils import average_feature_vector


path = './event_logs'
save_path = './encoding_results/word2vec'
os.makedirs(save_path, exist_ok=True)
for file in tqdm(sort_alphanumeric(os.listdir(path))):
    # read event log and import case id and labels
    ids, traces, y = retrieve_traces(read_log(path, file))

    start_time = time.time()

    # generate model
    model = Word2Vec(size=100, window=3, min_count=1, workers=-1)
    model = train_text_model(model, traces)

    # calculating the average feature vector for each sentence (trace)
    vectors = average_feature_vector(model, traces)

    end_time = time.time() - start_time

    # saving
    out_df = pd.DataFrame(vectors, columns=[f'feature_{i}' for i in range(100)])
    out_df['case'] = ids
    out_df['time'] = end_time
    out_df['label'] = y
    out_df.to_csv(f'{save_path}/{file}', index=False)
