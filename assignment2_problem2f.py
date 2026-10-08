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

def count_words_in_file(filename_queue,wordcount_queue,batch_size):
    """
    Counts the number of occurrences of words in the file
    Performs counting until a None is encountered in the queue
    Counts are stored in wordcount_queue
    Whitespace is ignored

    Parameters:
    - filename_queue, multiprocessing queue :  will contain filenames and None as a sentinel to indicate end of input
    - wordcount_queue, multiprocessing queue : (word,count) dictionaries are put in the queue, and end of input is indicated with a None
    - batch_size, int : size of batches to process

    Returns: None
    """
    worker_word_counts = {}
    batch_size_counter = 0
    while True:
        file = filename_queue.get()
        if file is None:                                    #if we encounter None, we put the remaining counts in the queue and then put a None to signal end of input to the merger, then we break out of the loop
            if worker_word_counts:
                wordcount_queue.put(worker_word_counts)
            wordcount_queue.put(None)
            break
        content_file = get_file(file)
        for word in content_file.split():
            if word in worker_word_counts:
                worker_word_counts[word] += 1
            else:
                worker_word_counts[word] = 1
        batch_size_counter += 1
        if batch_size_counter >= batch_size:
            wordcount_queue.put(worker_word_counts)
            worker_word_counts = {}
            batch_size_counter = 0





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



def merge_counts(out_queue,wordcount_queue,num_workers):
    """
    Merges the counts from the queue into the shared dict global_counts. 
    Quits when num_workers Nones have been encountered.

    Parameters:
    - global_counts, manager dict : global dictionary where to store the counts
    - wordcount_queue, manager queue : queue that contains (word,count) pairs and Nones to signal end of input from a worker
    - num_workers, int : number of workers (i.e., how many Nones to expect)

    Return value: None
    """
    global_counts = {}
    workers_done = 0
    # while workers are still running, we keep merging counts from the queue into global_counts. 
    while workers_done < num_workers:
        counts_to_merge = wordcount_queue.get()
        if counts_to_merge is None:
            workers_done += 1
        else:
            for (k,v) in counts_to_merge.items():
                if k not in global_counts:
                    global_counts[k] = v
                else:
                    global_counts[k] += v
    
    checksum = compute_checksum(global_counts)
    top10 = get_top10(global_counts)
    out_queue.put((top10, checksum))




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
    file_name_queue = mp.Queue()
    word_count_queue = mp.Queue()
    out_queue = mp.Queue()

    active_workers = 0
    workers = []
    while active_workers < num_workers:
        word_count_process = mp.Process(target=count_words_in_file, args=(file_name_queue,word_count_queue,batch_size))
        word_count_process.start()
        active_workers += 1
        workers.append(word_count_process)

    # merge process
    merger_process = mp.Process(target=merge_counts, args=(out_queue,word_count_queue,num_workers))
    merger_process.start()

    # put filenames in the queue
    for filename in get_filenames(path):
        file_name_queue.put(filename)

    # put None for each worker to signal end of input, as each worker will grab from queue until it encounters None
    for worker in workers:
        file_name_queue.put(None)

    # synchronize with merger process and get the output
    top10, checksum = out_queue.get()

    for worker_process in workers:
        worker_process.join()

    merger_process.join()

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f'Execution time: {execution_time:.2f} seconds')
    print(f'Checksum: {checksum}')
    print(f'Top 10 words: {top10}')

    # construct workers and queues
    # construct a special merger process
    # put filenames into the input queue
    # workers then put dictionaries for the merger
    # the merger shall return the checksum and top 10 through the out queue
    
