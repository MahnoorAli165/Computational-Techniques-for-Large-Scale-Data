import os
os.environ["PANDAS_BACKEND"] = "numpy"

import numpy as np
import argparse
import pandas as pd
import csv
import sys
import time
import cupy as cp
from cuvs.neighbors import brute_force


def cuvs_bruteforce_knn(X, Q, k=1, chunk_size=50000):
    """
    GPU brute-force KNN using CuPy with chunking to avoid OOM.
    X: (n, d)
    Q: (m, d)
    """

    X_gpu = cp.asarray(X, dtype=cp.float32)
    Q_gpu = cp.asarray(Q, dtype=cp.float32)

    n, d = X.shape
    m = Q.shape[0]

    # Precompute norms
    X_norm = cp.sum(X_gpu * X_gpu, axis=1).reshape(1, -1)  # (1, n)
    Q_norm = cp.sum(Q_gpu * Q_gpu, axis=1).reshape(-1, 1)  # (m, 1)

    # Output array
    best_idx = cp.zeros(m, dtype=cp.int64)
    best_dist = cp.full(m, cp.inf, dtype=cp.float32)

    # Process dataset in chunks
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)

        X_chunk = X_gpu[start:end]                     # (chunk, d)
        Xc_norm = X_norm[:, start:end]                 # (1, chunk)

        # Compute dot products
        dot = Q_gpu @ X_chunk.T                        # (m, chunk)

        # Compute distances
        dists = Q_norm + Xc_norm - 2 * dot             # (m, chunk)

        # Find best in this chunk
        idx_chunk = cp.argmin(dists, axis=1)
        dist_chunk = dists[cp.arange(m), idx_chunk]

        # Update global best
        mask = dist_chunk < best_dist
        best_dist[mask] = dist_chunk[mask]
        best_idx[mask] = idx_chunk[mask] + start

    return cp.asnumpy(best_idx)



def load_glove(fn):
    df = pd.read_table(fn, sep=' ', index_col=0, header=None,
                       quoting=csv.QUOTE_NONE, keep_default_na=False)
    X = np.ascontiguousarray(df, dtype=np.float32)
    L = df.index.tolist()
    return (X, L)


def load_pubs(fn):
    df = pd.read_csv(fn)
    L = df['name'].tolist()
    X = np.ascontiguousarray(df[['easting', 'northing']], dtype=np.float32)
    return (X, L)


def load_queries(fn):
    return np.loadtxt(fn, delimiter=' ', dtype=np.float32)


def load_query_labels(fn):
    with open(fn, 'r') as f:
        return f.read().splitlines()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CUVS brute-force nearest neighbor search')
    parser.add_argument('-d', '--dataset', type=str, required=True)
    parser.add_argument('-q', '--queries', type=str, required=True)
    parser.add_argument('-l', '--labels', type=str, required=False)
    args = parser.parse_args()

    # Load dataset
    t1 = time.time()
    if 'pubs' in args.dataset:
        (X, L) = load_pubs(args.dataset)
    elif 'glove' in args.dataset:
        (X, L) = load_glove(args.dataset)
    else:
        sys.stderr.write(f'{sys.argv[0]}: error: Only glove/pubs supported\n')
        exit(1)
    t2 = time.time()

    (n, d) = X.shape
    assert len(L) == n

    # Load queries
    t3 = time.time()
    Q = load_queries(args.queries)
    t4 = time.time()

    assert X.flags['C_CONTIGUOUS']
    assert Q.flags['C_CONTIGUOUS']
    assert X.dtype == np.float32
    assert Q.dtype == np.float32

    m = Q.shape[0]
    assert Q.shape[1] == d

    # Load labels (optional)
    t5 = time.time()
    QL = None
    if args.labels is not None:
        QL = load_query_labels(args.labels)
        assert len(QL) == m
    t6 = time.time()

    # Run CUVS brute-force KNN
    print("Using CUVS brute-force KNN")
    t_start = time.time()
    I = cuvs_bruteforce_knn(X, Q, k=1)
    cp.cuda.Stream.null.synchronize()
    t_end = time.time()

    runtime = t_end - t_start
    throughput = m / runtime

    print(f'Runtime: {runtime:.4f} seconds')
    print(f'Throughput: {throughput:.2f} queries/second')

    # Compute errors
    num_erroneous = 0
    if QL is not None:
        for (i, j) in enumerate(I):
            if QL[i] != L[j]:
                num_erroneous += 1

    print(f'Incorrect nearest neighbors: {num_erroneous}')

    print(f'Loading dataset ({n} vectors of length {d}) took', t2 - t1)
    print(f'Performing {m} NN queries took {runtime:.4f} seconds')
    print(f'Number of erroneous queries: {num_erroneous}')
