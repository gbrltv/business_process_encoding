import os
import time
import pandas as pd
from tqdm import tqdm
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from skmultiflow.utils import calculate_object_size
from utils import read_log
from utils import sort_alphanumeric
from utils import retrieve_traces
from utils import train_text_model
from utils import average_feature_vector_doc2vec


path = './event_logs'
save_path = './encoding_results/doc2vec'
os.makedirs(save_path, exist_ok=True)
for file in tqdm(sort_alphanumeric(os.listdir(path))):
    # read event log and import case id and labels
    ids, traces, y = retrieve_traces(read_log(path, file))
    tagged_traces = [TaggedDocument(words=act, tags=[str(i)]) for i, act in enumerate(traces)]

    start_time = time.time()

    # generate model
    model = Doc2Vec(vector_size=100, min_count=1, window=3, dm=1, workers=-1)
    model.build_vocab(tagged_traces)

    # calculating the average feature vector for each sentence (trace)
    vectors = average_feature_vector_doc2vec(model, traces)

    end_time = time.time() - start_time
    memory = calculate_object_size(vectors)

    # saving
    out_df = pd.DataFrame(vectors, columns=[f'feature_{i}' for i in range(100)])
    out_df['case'] = ids
    out_df['time'] = end_time
    out_df['memory'] = memory
    out_df['label'] = y
    out_df.to_csv(f'{save_path}/{file}', index=False)
