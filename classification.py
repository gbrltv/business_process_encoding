import os
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from skmultiflow.utils import calculate_object_size
from utils import sort_alphanumeric


def read_data(path: str, file: str):
    """
    Reads file containing the encodings for a given event log

    Parameters
    -----------------------
    path: str,
        File path
    log: str,
        File name
    Returns
    -----------------------
    The encoding vectors and their corresponding labels
    """
    df = pd.read_csv(f'{path}/{file}')
    y = list(df['label'])
    del df['case'], df['time'], df['label']
    vectors = df.to_numpy()
    del df

    return vectors, y

def compute_memory(obj1, obj2, obj3):
    """
    Calculates the allocated memory of three objects (rf, encodings and labels)

    Returns
    -----------------------
    Returns an appropriate format to write a csv file
    """
    rf_size = calculate_object_size(obj1)
    encoding_size = calculate_object_size(obj2)
    label_size = calculate_object_size(obj3)
    total_size = rf_size + encoding_size + label_size

    return [rf_size, encoding_size, label_size, total_size]

def compute_metrics(y_true, y_pred, binary=True):
    """
    Computes classification metrics

    Parameters
    -----------------------
    y_true,
        List of true instance labels
    y_pred,
        List of predicted instance labels
    binary,
        Controls the computation of binary only metrics
    Returns
    -----------------------
    Classification metrics
    """
    accuracy = metrics.accuracy_score(y_true, y_pred)
    accuracy_balanced = metrics.balanced_accuracy_score(y_true, y_pred)
    f1_micro = metrics.f1_score(y_true, y_pred, average='micro')
    f1_macro = metrics.f1_score(y_true, y_pred, average='macro')
    f1_weighted = metrics.f1_score(y_true, y_pred, average='weighted')
    precision_micro = metrics.precision_score(y_true, y_pred, average='micro', zero_division=0)
    precision_macro = metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
    precision_weighted = metrics.precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_micro = metrics.recall_score(y_true, y_pred, average='micro', zero_division=0)
    recall_macro = metrics.recall_score(y_true, y_pred, average='macro', zero_division=0)
    recall_weighted = metrics.recall_score(y_true, y_pred, average='weighted', zero_division=0)

    if binary:
        f1_binary = metrics.f1_score(y_true, y_pred, average='binary', pos_label='normal')
        precision_binary = metrics.precision_score(y_true, y_pred, average='binary', pos_label='normal')
        recall_binary = metrics.recall_score(y_true, y_pred, average='binary', pos_label='normal')
    else:
        f1_binary = np.nan
        precision_binary = np.nan
        recall_binary = np.nan

    return [accuracy, accuracy_balanced, f1_binary, f1_micro,
    f1_macro, f1_weighted, precision_binary, precision_micro,
    precision_macro, precision_weighted, recall_binary,
    recall_micro, recall_macro, recall_weighted]


encodings = sort_alphanumeric(os.listdir('encoding_results'))
files = sort_alphanumeric(os.listdir('encoding_results/alignment'))

out = []
for file in tqdm(files, total=len(files), desc='Calculating classification metrics'):
    for encoding in encodings:
        path = f'encoding_results/{encoding}'
        vectors, labels = read_data(path, file)
        metrics_list = []
        for i in range(30):
            start_time = time.time()
            X_train, X_test, y_train, y_test = train_test_split(vectors, labels, test_size=0.2)

            rf = RandomForestClassifier(n_jobs=-1).fit(X_train, y_train)
            y_pred = rf.predict(X_test)

            if 'all' in file:
                metrics_ = compute_metrics(y_test, y_pred, False)
            else:
                metrics_ = compute_metrics(y_test, y_pred)

            end_time = time.time() - start_time
            metrics_.extend([end_time])
            metrics_.extend(compute_memory(rf, vectors, labels))
            metrics_list.append(metrics_)

        metrics_array = np.array(metrics_list)
        metrics_array = list(np.nanmean(metrics_array, axis=0))
        file_encoding = [file.split('.csv')[0], encoding]
        file_encoding.extend(metrics_array)
        out.append(file_encoding)

columns = ['log', 'encoding', 'accuracy', 'accuracy_balanced', 'f1_binary',
    'f1_micro', 'f1_macro', 'f1_weighted', 'precision_binary', 'precision_micro',
    'precision_macro', 'precision_weighted', 'recall_binary', 'recall_micro',
    'recall_macro', 'recall_weighted', 'time', 'mem_rf', 'mem_encoding',
    'mem_labels', 'mem_total']

pd.DataFrame(out, columns=columns).to_csv('results.csv', index=False)
