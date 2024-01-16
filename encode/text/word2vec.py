import time
import pandas as pd
from gensim.models import Word2Vec
from .utils import retrieve_traces
from .utils import retrieve_encoding


def run_word2vec(config, log):
    ids, traces = retrieve_traces(log)

    start_time = time.time()

    # generate model
    model = Word2Vec(vector_size=config["vector_size"], window=3, min_count=1, sg=0, workers=-1)
    model.build_vocab(traces)
    model.train(traces, total_examples=len(traces), epochs=10)

    # calculating the feature vector for each sentence (trace)
    vectors = retrieve_encoding(model, traces, config["aggregation"])

    end_time = time.time() - start_time
    print(f"\nWord2vec took {round(end_time, 2)} seconds")

    # saving
    encoded_df = pd.DataFrame(vectors, columns=[f'{i}' for i in range(config["vector_size"])])
    encoded_df.insert(0, "case", ids)

    return encoded_df
