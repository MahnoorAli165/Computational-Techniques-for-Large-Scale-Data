#!/usr/bin/env python3
import argparse
import sys
import os
import math
import time
from pyspark import SparkContext, SparkConf

# Use your exact implementations
from assignment5_problem1 import murmur3_32
from assignment5_problem2 import compute_jr


def get_files(path):
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(".txt"):
                yield os.path.join(root, f)


def alpha_m(m):
    if m == 16:
        return 0.673
    elif m == 32:
        return 0.697
    elif m == 64:
        return 0.709
    else:
        return 0.7213 / (1 + 1.079 / m)


def hll(words, m, seed):
    registers = [0] * m
    log2m = int(math.log2(m))

    for w in words:
        # compute_jr does BOTH hashing and j/r extraction
        j, r = compute_jr(w, seed, log2m)
        registers[j] = max(registers[j], r)

    return registers


def merge_registers(a, b):
    return [max(x, y) for x, y in zip(a, b)]


def estimate_cardinality(registers, m):
    alpha = alpha_m(m)
    Z = sum(2 ** -r for r in registers)
    E = alpha * m * m / Z
    return E


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to Gutenberg dataset")
    parser.add_argument("-m", type=int, default=128, help="Number of registers")
    parser.add_argument("-s", type=lambda x: int(x, 16), default=0x00000000, help="Seed")
    parser.add_argument("-w", type=int, default=4, help="Number of workers")
    args = parser.parse_args()

    if not os.path.isdir(args.path):
        sys.stderr.write(f"{sys.argv[0]}: `{args.path}` is not a valid directory\n")
        sys.exit(1)

    conf = SparkConf().setAppName("HyperLogLog").setMaster(f"local[{args.w}]")
    sc = SparkContext(conf=conf)

    start = time.time()

    files = list(get_files(args.path))
    rdd = sc.parallelize(files, args.w)

    registers = (
        rdd.flatMap(lambda f: open(f, "r", encoding="utf-8", errors="ignore").read().split())
           .mapPartitions(lambda words: [hll(words, args.m, args.s)])
           .reduce(merge_registers)
    )

    estimate = estimate_cardinality(registers, args.m)
    end = time.time()

    print(f"Cardinality estimate: {estimate}")
    print(f"Number of workers: {args.w}")
    print(f"Took {end - start:.4f} s")

    sc.stop()


if __name__ == "__main__":
    main()
