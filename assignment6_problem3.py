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
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms

def construct_queries(queries_fn, word_to_idx, X):
    """
    Reads queries (one string per line) and returns:
    - The query vectors as a matrix Q (one query per row)
    - Query labels as a list of strings
    """
    with open(queries_fn, 'r') as f:
        queries = f.read().splitlines()
    Q = np.zeros((len(queries), X.shape[1]))
    for i in range(len(queries)):
        Q[i,:] = X[word_to_idx[queries[i]],:]
    return (Q,queries)

class RandomHyperplanes:
    """
    This class mimics the interface of sklearn:
    - the constructor sets the number of hyperplanes
    - the random hyperplanes are drawn when fit() is called 
      (input dimension is set)
    - transform actually transforms the vectors
    - fit_transform does fit first, followed by transform
    """
    def __init__(self, D, seed = None)->None:
        """
        Sets the number of hyperplanes (D) and the optional random number seed
        """
        self._D = D
        self._seed = seed

    def fit(self, X):
        rng = np.random.default_rng(self._seed)
        d = X.shape[1]

        # Draw D random Gaussian vectors of dimension d
        R = rng.normal(loc=0.0, scale=1.0, size=(self._D, d))

        # Normalize each row to unit length
        norms = np.linalg.norm(R, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        R = R / norms

        # Store as attribute
        self._R = R
        return self


    def transform(self, X):
        # Project: X' = X R^T
        X_proj = X @ self._R.T

        # Threshold: > 0 → 1, else 0
        X_bin = (X_proj > 0).astype(np.int8)

        return X_bin


    def fit_transform(self, X):
        """
        Calls fit() followed by transform()
        """
        self.fit(X)
        return self.transform(X)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', help='Random hyperplanes dimension', type=int,
                            required = True)
    parser.add_argument('dataset', help='Glove dataset filename',
                            type=str)
    parser.add_argument('queries', help='Queries filename', type=str)
    args = parser.parse_args()
    
    (word_to_idx, idx_to_word, X) = load_glove(args.dataset)


    X = normalize(X)

    (Q,queries) = construct_queries(args.queries, word_to_idx, X)

    start = time.time()

    rh = RandomHyperplanes(args.D, 1234)
    X2 = rh.fit_transform(X)
    Q2 = rh.transform(Q)

    end = time.time()

    print(end-start)
