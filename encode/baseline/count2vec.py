import time
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from .utils import extract_corpus


def run_count2vec(config, log):
    ids, traces = extract_corpus(log)

    start_time = time.time()

    # count2vec encoding
    model = CountVectorizer()
    encoding = model.fit_transform(traces)

    end_time = time.time() - start_time
    print(f"\nCount2vec took {round(end_time, 2)} seconds")

    # formatting
    encoded_df = pd.DataFrame(encoding.toarray(), columns=[f"{i}" for i in range(encoding.toarray().shape[1])])
    encoded_df.insert(0, "case", ids)

    return encoded_df
