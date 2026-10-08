import numpy as np
import argparse
import pandas as pd
import csv
import sys
import time
import cupy as cp

def gpu_linear_scan(X, Q, batch_size=None, chunk_size=50000):
    """
    Fully GPU-based linear scan with two-level batching:
    - batch over queries (b)
    - chunk over dataset (c)
    This avoids allocating a (b × n) matrix and prevents OOM.
    """

    # Move data to GPU
    X_gpu = cp.asarray(X, dtype=cp.float32)
    Q_gpu = cp.asarray(Q, dtype=cp.float32)

    n, d = X_gpu.shape
    m, _ = Q_gpu.shape

    # Precompute norms
    X_norm = cp.sum(X_gpu ** 2, axis=1)  # (n,)

    if batch_size is None:
        batch_size = m

    indices = []

    for qs in range(0, m, batch_size):
        qe = min(qs + batch_size, m)
        Qb = Q_gpu[qs:qe]                 # (b, d)
        Q_norm = cp.sum(Qb ** 2, axis=1)  # (b,)

        b = Qb.shape[0]

        # Running best distances and indices
        best_dist = cp.full((b,), cp.inf, dtype=cp.float32)
        best_idx  = cp.full((b,), -1, dtype=cp.int64)

        # Chunk over dataset X
        for xs in range(0, n, chunk_size):
            xe = min(xs + chunk_size, n)

            Xc = X_gpu[xs:xe]                 # (c, d)
            Xc_norm = X_norm[xs:xe]           # (c,)

            # Compute dot products for this chunk
            dot = Qb @ Xc.T                   # (b, c)
            D2_chunk = Q_norm[:, None] + Xc_norm[None, :] - 2.0 * dot

            # Local minima inside this chunk
            local_min = cp.min(D2_chunk, axis=1)
            local_arg = cp.argmin(D2_chunk, axis=1)

            # Update global minima
            mask = local_min < best_dist
            best_dist[mask] = local_min[mask]
            best_idx[mask]  = xs + local_arg[mask]

        indices.append(cp.asnumpy(best_idx))

    return np.concatenate(indices, axis=0)


def load_glove(fn):
    df = pd.read_table(fn, sep=' ', index_col=0, header=None,
                       quoting=csv.QUOTE_NONE, keep_default_na=False)
    X = np.ascontiguousarray(df, dtype=np.float32)
    L = df.index.tolist()
    return (X, L)

def load_pubs(fn):
    df = pd.read_csv(fn)
    L = df['name'].tolist()
    X = np.ascontiguousarray(df[['easting','northing']], dtype=np.float32)
    return (X, L)

def load_queries(fn):
    return np.loadtxt(fn, delimiter=' ', dtype=np.float32)

def load_query_labels(fn):
    with open(fn,'r') as f:
        return f.read().splitlines()

def gpu_available():
    try:
        n = cp.cuda.runtime.getDeviceCount()
        return n > 0
    except cp.cuda.runtime.CUDARuntimeError:
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='GPU-only NN search with batching and chunking')
    parser.add_argument('-d', '--dataset', type=str, required=True)
    parser.add_argument('-q', '--queries', type=str, required=True)
    parser.add_argument('-l', '--labels', type=str, required=False)
    parser.add_argument('-b', '--batch-size', type=int, required=False)
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

    # GPU-only execution
    print("Using GPU (CuPy) for NN search")
    t_start = time.time()
    I = gpu_linear_scan(X, Q, batch_size=args.batch_size or 1024)
    cp.cuda.Stream.null.synchronize()
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
