#!/usr/bin/env python3

import numpy as np
import pandas as pd
import csv
import argparse
import time

def load_glove(filename):
    """
    Loads the glove dataset. Returns three things:
    A dictionary that contains a map from words to rows in the dataset.
    A reverse dictionary that maps rows to words.
    The embeddings dataset as a NumPy array.
    """
    df = pd.read_table(filename, sep=' ', index_col=0, header=None,
                           quoting=csv.QUOTE_NONE)
    word_to_idx = dict()
    idx_to_word = dict()
    for (i,word) in enumerate(df.index):
        word_to_idx[word] = i
        idx_to_word[i] = word
    return (word_to_idx, idx_to_word, df.to_numpy())

def normalize(X):
    # Compute L2 norms for each row: shape (n, 1)
    norms = np.linalg.norm(X, axis=1, keepdims=True)

    # Avoid division by zero (rare but safe)
    norms[norms == 0] = 1.0

    # Divide each row by its norm
    return X / norms

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='Glove dataset filename',
                            type=str)
    args = parser.parse_args()
    (word_to_idx, idx_to_word, X) = load_glove(args.dataset)

    start = time.time()

    X = normalize(X)

    end = time.time()

    print(end - start)
