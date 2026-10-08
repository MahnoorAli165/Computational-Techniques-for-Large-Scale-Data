import os
import argparse
import sys
import time
import multiprocessing as mp


def get_filenames(path):
    """
    A generator function: Iterates through all .txt files in the path and
    returns the full names of the files

    Parameters:
    - path : string, path to walk through

    Yields:
    The full filenames of all files ending in .txt
    """
    for (root, dirs, files) in os.walk(path):
        for file in files:
            if file.endswith('.txt'):
                yield f'{root}/{file}'

def get_file(path):
    """
    Reads the content of the file and returns it as a string.

    Parameters:
    - path : string, path to a file

    Return value:
    The content of the file in a string.
    """
    with open(path,'r') as f:
        return f.read()

def count_words_in_file(file):
    """
    Counts the number of occurrences of words in the file
    Whitespace is ignored

    Parameters:
    - file, string : the content of a file

    Returns: Dictionary that maps words (strings) to counts (ints)
    """
    counts = dict()
    for word in file.split():
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1
    return counts



def get_top10(counts):
    """
    Determines the 10 words with the most occurrences.
    Ties can be solved arbitrarily.

    Parameters:
    - counts, dictionary : a mapping from words (str) to counts (int)
    
    Return value:
    A list of (count,word) pairs (int,str)
    """
    top10 = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
    top10 = [(count, word) for (word, count) in top10]
    return top10



def merge_counts(dict_to, dict_from):
    """
    Merges the word counts from dict_from into dict_to, such that
    if the word exists in dict_to, then the count is added to it,
    otherwise a new entry is created with count from dict_from

    Parameters:
    - dict_to, dictionary : dictionary to merge to
    - dict_from, dictionary : dictionary to merge from

    Return value: None
    """
    for (k,v) in dict_from.items():
        if k not in dict_to:
            dict_to[k] = v
        else:
            dict_to[k] += v



def compute_checksum(counts):
    """
    Computes the checksum for the counts as follows:
    The checksum is the sum of products of the length of the word and its count

    Parameters:
    - counts, dictionary : word to count dictionary

    Return value:
    The checksum (int)
    """
    checksum = 0
    for word, count in counts.items():
        checksum += len(word) * count
    return checksum
   # raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Counts words of all the text files in the given directory')
    parser.add_argument('-w', '--num-workers', help = 'Number of workers', default=1, type=int)
    parser.add_argument('-b', '--batch-size', help = 'Batch size', default=1, type=int)
    parser.add_argument('path', help = 'Path that contains text files')
    args = parser.parse_args()

    path = args.path

    if not os.path.isdir(path):
        sys.stderr.write(f'{sys.argv[0]}: ERROR: `{path}\' is not a valid directory!\n')
        quit(1)

    num_workers = args.num_workers
    if num_workers < 1:
        sys.stderr.write(f'{sys.argv[0]}: ERROR: Number of workers must be positive (got {num_workers})!\n')
        quit(1)

    batch_size = args.batch_size
    if batch_size < 1:
        sys.stderr.write(f'{sys.argv[0]}: ERROR: Batch size must be positive (got {batch_size})!\n')
        quit(1)

    start_time = time.perf_counter()
    files = [get_file(fn) for fn in get_filenames(path)]
    
    with mp.Pool(processes=num_workers) as pool:
        file_counts = pool.map(count_words_in_file, files)    


    global_counts = dict()
    for counts in file_counts:
        merge_counts(global_counts,counts)

    time_4 = time.perf_counter()

    #Time measurements
    total_time = time_4 - start_time
    """
    Container did not have matplotlib installed, so I ran this part separately on my local machine:

    workers = [1,2,4,8,16,32,64]
    times = [144.24 , 65.93 , 47.54 , 42.24 , 39.78 , 41.04 , 43.52]
    speedups = []
    theoretical_speedups = 2.71
    for t in times:
        speedup = 69.77 / t
        speedups.append(speedup)

    plt.figure(figsize=(10, 6))
    plt.plot(workers, speedups, marker='o', label='Actual Speedup')
    plt.axhline(y=theoretical_speedups, color='r', linestyle='--', label='Theoretical Speedup (1.30)')
    plt.xscale('log', basex=2)
    plt.xticks(workers)
    plt.xlabel('Number of Workers')
    plt.ylabel('Speedup')
    plt.title('Speedup vs Number of Workers')
    plt.legend()
    plt.grid()
    """

    checksum = compute_checksum(global_counts)
    print(f'Checksum: {checksum}') 
    print(f'Total time: {total_time:.2f} seconds')
 
