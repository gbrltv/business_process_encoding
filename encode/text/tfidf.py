import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from .utils import extract_corpus


def run_tfidf(config, log):
    ids, traces = extract_corpus(log)

    start_time = time.time()

    # generate model
    model = TfidfVectorizer()
    encoding = model.fit_transform(traces)

    end_time = time.time() - start_time
    print(f"\nTFIDF took {round(end_time, 2)} seconds")

    # saving
    encoded_df = pd.DataFrame(encoding.toarray(), columns=[f"{i}" for i in range(encoding.toarray().shape[1])])
    encoded_df.insert(0, "case", ids)

    return encoded_df
