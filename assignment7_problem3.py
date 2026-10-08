import numpy as np
import argparse
import pandas as pd
import csv
import sys
import time

def linear_scan(X, Q, chunk_size=None):
    """
    Memory-safe matrix-multiplication NN search.
    Computes distances in chunks to avoid allocating (n × m) matrices.
    """

    # If user did not pass -b, use default chunk size
    if chunk_size is None:
        chunk_size = 50000

    X = np.asarray(X, dtype=np.float32)
    Q = np.asarray(Q, dtype=np.float32)

    n, d = X.shape
    m, _ = Q.shape

    X_norm = np.sum(X * X, axis=1)        # (n,)
    Q_norm = np.sum(Q * Q, axis=1)        # (m,)

    best_dist = np.full((m,), np.inf, dtype=np.float32)
    best_idx  = np.full((m,), -1, dtype=np.int64)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)

        Xc = X[start:end]                 # (c, d)
        Xc_norm = X_norm[start:end]       # (c,)

        # Compute inner products for this chunk
        inner = Xc @ Q.T                  # (c, m)

        # Compute distances
        D2 = Xc_norm[:, None] + Q_norm[None, :] - 2.0 * inner

        # Find best in this chunk
        local_min = np.min(D2, axis=0)
        local_arg = np.argmin(D2, axis=0)

        mask = local_min < best_dist
        best_dist[mask] = local_min[mask]
        best_idx[mask]  = start + local_arg[mask]

    return best_idx



def load_glove(fn):
    """
    Loads the glove dataset from the file
    Returns (X,L) where X is the dataset vectors and L is the words associated
    with the respective rows.
    """
    df = pd.read_table(fn, sep = ' ', index_col = 0, header = None,
                           quoting = csv.QUOTE_NONE, keep_default_na = False)
    X = np.ascontiguousarray(df, dtype = np.float32)
    L = df.index.tolist()
    return (X, L)

def load_pubs(fn):
    """
    Loads the pubs dataset from the file
    Returns (X,L) where X is the dataset vectors (easting,northing) and 
    L is the list of names of pubs, associated with each row
    """
    df = pd.read_csv(fn)
    L = df['name'].tolist()
    X = np.ascontiguousarray(df[['easting','northing']], dtype = np.float32)
    return (X, L)

def load_queries(fn):
    """
    Loads the m*d array of query vectors from the file
    """
    return np.loadtxt(fn, delimiter = ' ', dtype = np.float32)

def load_query_labels(fn):
    """
    Loads the m-long list of correct query labels from a file
    """
    with open(fn,'r') as f:
        return f.read().splitlines()

if __name__ == '__main__':
    parser = argparse.ArgumentParser( \
          description = 'Perform nearest neighbor queries under the '
          'Euclidean metric using linear scan, measure the time '
          'and optionally verify the correctness of the results')
    parser.add_argument(
        '-d', '--dataset', type=str, required=True,
        help = 'Dataset file (must be pubs or glove)')
    parser.add_argument(
        '-q', '--queries', type=str, required=True,
        help = 'Queries file (must be compatible with the dataset)')
    parser.add_argument(
        '-l', '--labels', type=str, required=False,
        help = 'Optional correct query labels; if provided, the correctness '
        'of returned results is checked')
    parser.add_argument(
        '-b', '--batch-size', type=int, required=False,
        help = 'Size of batches')
    args = parser.parse_args()

    t1 = time.time()
    if 'pubs' in args.dataset:
        (X,L) = load_pubs(args.dataset)
    elif 'glove' in args.dataset:
        (X,L) = load_glove(args.dataset)
    else:
        sys.stderr.write(f'{sys.argv[0]}: error: Only glove/pubs supported\n')
        exit(1)
    t2 = time.time()

    (n,d) = X.shape
    assert len(L) == n

    t3 = time.time()
    Q = load_queries(args.queries)
    t4 = time.time()

    assert X.flags['C_CONTIGUOUS']
    assert Q.flags['C_CONTIGUOUS']
    assert X.dtype == np.float32
    assert Q.dtype == np.float32
    
    m = Q.shape[0]
    assert Q.shape[1] == d

    t5 = time.time()
    QL = None
    if args.labels is not None:
        QL = load_query_labels(args.labels)
        assert len(QL) == m
    t6 = time.time()

   # Run NN search
    t_start = time.time()
    I = linear_scan(X, Q, chunk_size=args.batch_size)
    t_end = time.time()

    runtime = t_end - t_start
    throughput = m / runtime

    print(f'Runtime: {runtime:.4f} seconds')
    print(f'Throughput: {throughput:.2f} queries/second')



    num_erroneous = 0
    if QL is not None:
        for (i, j) in enumerate(I):
            if QL[i] != L[j]:
                num_erroneous += 1

    print(f'Incorrect nearest neighbors: {num_erroneous}')


    print(f'Loading dataset ({n} vectors of length {d}) took', t2-t1)
    print(f'Performing {m} NN queries took {runtime:.4f} seconds')
    print(f'Number of erroneous queries: {num_erroneous}')
