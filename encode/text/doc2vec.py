import time
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from .utils import retrieve_traces


def run_doc2vec(config, log):
    ids, traces = retrieve_traces(log)

    start_time = time.time()

    tagged_traces = [TaggedDocument(words=act, tags=[str(i)]) for i, act in enumerate(traces)]

    # generate model
    model = Doc2Vec(vector_size=config["vector_size"], min_count=1, window=3, dm=1, workers=-1)
    model.build_vocab(tagged_traces)
    vectors = [model.infer_vector(trace) for trace in traces]

    end_time = time.time() - start_time
    print(f"\nDoc2vec took {round(end_time, 2)} seconds")

    # saving
    encoded_df = pd.DataFrame(vectors, columns=[f'{i}' for i in range(config["vector_size"])])
    encoded_df.insert(0, "case", ids)

    return encoded_df
