import time
import pandas as pd
from sklearn.preprocessing import Binarizer
from sklearn.feature_extraction.text import CountVectorizer
from .utils import extract_corpus



def run_onehot(config, log):
    ids, traces = extract_corpus(log)

    start_time = time.time()

    # onehot encoding
    corpus = CountVectorizer(analyzer="word").fit_transform(traces)
    encoding = Binarizer().fit_transform(corpus.toarray())

    end_time = time.time() - start_time
    print(f"\nOne-hot encoding took {round(end_time, 2)} seconds")

    # formatting
    encoded_df = pd.DataFrame(encoding, columns=[f"{i}" for i in range(encoding.shape[1])])
    encoded_df.insert(0, "case", ids)

    return encoded_df
