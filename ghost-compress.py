import numpy as np
import sys
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import pickle
import gc
import os

def extract_subsequences_chunk(data, min_length, max_length, start, end):
    subsequences = defaultdict(int)
    for length in range(min_length, max_length + 1):
        for i in range(start, end - length + 1):
            subseq = tuple(data[i:i + length])
            subsequences[subseq] += 1
    return subsequences

def merge_dicts(dicts):
    result = defaultdict(int)
    for d in dicts:
        for key, value in d.items():
            result[key] += value
    return result

def filter_subsequences(subsequences):
    # Remove subsequences that occur only once
    return {k: v for k, v in subsequences.items() if v > 1}

def extract_and_filter_subsequences(data, min_length=1, max_length=256):
    data_length = len(data)
    num_workers = cpu_count()
    chunk_size = data_length // num_workers

    pool = Pool(processes=num_workers)
    tasks = []
    for i in range(num_workers):
        start = i * chunk_size
        end = data_length if i == num_workers - 1 else (i + 1) * chunk_size
        tasks.append((data, min_length, max_length, start, end))

    subsequence_chunks = pool.starmap(extract_subsequences_chunk, tasks)
    pool.close()
    pool.join()

    subsequences = merge_dicts(subsequence_chunks)
    filtered_subsequences = filter_subsequences(subsequences)
    return filtered_subsequences

def find_most_common_subsequences(subsequence_counts, top_n=256):
    scored_subsequences = {k: (v * len(k), v) for k, v in subsequence_counts.items()}
    sorted_subsequences = sorted(scored_subsequences.items(), key=lambda item: item[1][0], reverse=True)
    return sorted_subsequences[:top_n]

def find_missing_sequences_chunk(data, sequence_length, start, end):
    present_sequences = set()
    for i in range(start, end - sequence_length + 1):
        subseq = tuple(data[i:i + sequence_length])
        present_sequences.add(subseq)
    return present_sequences

def find_missing_sequences(data, sequence_length):
    data_length = len(data)
    num_workers = cpu_count()
    chunk_size = data_length // num_workers

    pool = Pool(processes=num_workers)
    tasks = []
    for i in range(num_workers):
        start = i * chunk_size
        end = data_length if i == num_workers - 1 else (i + 1) * chunk_size
        tasks.append((data, sequence_length, start, end))

    present_sequence_chunks = pool.starmap(find_missing_sequences_chunk, tasks)
    pool.close()
    pool.join()

    present_sequences = set().union(*present_sequence_chunks)

    all_possible_sequences = set()
    for i in range(256**sequence_length):
        sequence = tuple((i >> (8 * j)) & 0xFF for j in range(sequence_length))
        all_possible_sequences.add(sequence)

    missing_sequences = all_possible_sequences - present_sequences
    return sorted(missing_sequences)

def read_file(file_path):
    """
    Read binary data from a file.
    """
    with open(file_path, 'rb') as f:
        data = f.read()
    return data

def write_file(file_path, data):
    """
    Write binary data to a file.
    """
    with open(file_path, 'wb') as f:
        f.write(data)

def append_to_dict_file(dict_file_path, dictionary):
    """
    Append the dictionary mappings to the dictionary file.
    """
    with open(dict_file_path, 'ab') as dict_file:
        for key, value in dictionary.items():
            pickle.dump((key, value), dict_file)

def get_combined_size(file1, file2):
    """
    Get the combined size of two files.
    """
    return os.path.getsize(file1) + os.path.getsize(file2)

def get_iteration_count(dict_file_path):
    """
    Get the number of iterations from the dictionary file.
    """
    if not os.path.exists(dict_file_path):
        return None
    with open(dict_file_path, 'rb') as dict_file:
        iterations = 0
        while True:
            try:
                pickle.load(dict_file)
                iterations += 1
            except EOFError:
                break
    return iterations

def main(file_path, total_iterations, max_length, top_n=256):
    dict_file_path = file_path + '.dict'
    compressed_file_path = file_path + '.boo'
    compressed = False

    if file_path.endswith('.boo'):
        compressed_file_path = file_path
        dict_file_path = file_path.replace('.boo','.dict')
        iterations_done = get_iteration_count(dict_file_path)
        if iterations_done is None:
            print("Dictionary file not found. Cannot continue compression.")
            sys.exit(1)
        data = read_file(file_path)
        compressed = True
        iteration_count = iterations_done
    else:
        iteration_count = 0
        iterations_done = 0
        data = read_file(file_path)
        # Initialize or clear the dictionary file
        with open(dict_file_path, 'wb') as dict_file:
            pass

    # Initial combined size
    if not compressed:
        write_file(compressed_file_path, data)
    previous_combined_size = get_combined_size(dict_file_path, compressed_file_path)
    
    # Introduce bias to avoid local maximum
    previous_combined_size *= 1.01

    sequence_length = 1

    while sequence_length < max_length and (total_iterations == -1 or iteration_count - iterations_done < total_iterations):
        print(f"Processing sequence length: {sequence_length}")
        missing_sequences = find_missing_sequences(data, sequence_length)
        if not missing_sequences:
            sequence_length += 1
            continue

        while missing_sequences and (total_iterations == -1 or iteration_count < total_iterations):
            subsequence_counts = extract_and_filter_subsequences(data, sequence_length + 1, max_length)
            most_common_subsequences = find_most_common_subsequences(subsequence_counts, top_n)

            if not most_common_subsequences:
                break

            highest_score_sequence, (score, occurrences) = most_common_subsequences[0]

            # Use the first missing sequence and then remove it from the list
            first_missing_sequence = missing_sequences.pop(0)

            # Replace the most common sequence with the missing sequence
            data = data.replace(bytes(highest_score_sequence), bytes(first_missing_sequence))
            current_mapping = {first_missing_sequence: highest_score_sequence}

            # Calculate combined size
            current_combined_size = len(first_missing_sequence) + len(data) + os.path.getsize(dict_file_path)
            if current_combined_size > previous_combined_size:
                print("Max compression reached. Stopping compression.")
                return

            # Save the current mapping to the dictionary file
            append_to_dict_file(dict_file_path, current_mapping)

            # Save the updated compressed file
            write_file(compressed_file_path, data)

            # Trigger garbage collection to free up memory
            gc.collect()

            previous_combined_size = min(current_combined_size, previous_combined_size)

            iteration_count += 1

            # Print progress
            print(f"Iteration {iteration_count} for sequence length {sequence_length} completed.")

        sequence_length += 1

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python ghost-compress.py <file_path> <iterations> <max_length>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    total_iterations = int(sys.argv[2])
    max_length = int(sys.argv[3])
    top_n = 256  # Number of top common subsequences to print
    main(file_path, total_iterations, max_length, top_n)
