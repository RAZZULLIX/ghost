import numpy as np
import sys
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import gc
import os
from datetime import datetime, timedelta

def time_difference(start_time, end_time):
    diff = end_time - start_time
    if diff < timedelta(0):
        diff = -diff
    total_seconds = int(diff.total_seconds())
    milliseconds = int((diff.total_seconds() - total_seconds) * 1000)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    if hours > 99:
        return "[99:59:59.999]+"
    else:
        return f"[{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}]"

def merge_dicts(dicts):
    result = defaultdict(int)
    for d in dicts:
        for key, value in d.items():
            result[key] += value
    return result

def filter_subsequences(subsequences):
    return {k: v for k, v in subsequences.items() if v > 1}

def extract_and_filter_subsequences(data, min_length=1, max_length=256):
    data_length = len(data)
    num_workers = cpu_count()
    chunk_size = data_length // num_workers

    with Pool(processes=num_workers) as pool:
        tasks = [(data, min_length, max_length, i * chunk_size, data_length if i == num_workers - 1 else (i + 1) * chunk_size) for i in range(num_workers)]
        subsequence_chunks = pool.starmap(extract_subsequences_chunk, tasks)

    subsequences = merge_dicts(subsequence_chunks)
    filtered_subsequences = filter_subsequences(subsequences)
    return filtered_subsequences

def extract_subsequences_chunk(data, min_length, max_length, start, end):
    subsequences = defaultdict(int)
    for length in range(min_length, max_length + 1):
        for i in range(start, end - length + 1):
            subseq = data[i:i + length]
            subsequences[subseq] += 1
    return subsequences

def find_most_common_subsequences(subsequence_counts, missing_sequence_length, top_n=256):
    scored_subsequences = {k: ((len(k) - missing_sequence_length) * v, v) for k, v in subsequence_counts.items()}
    sorted_subsequences = sorted(scored_subsequences.items(), key=lambda item: item[1][0], reverse=True)
    return sorted_subsequences[:top_n]

def find_missing_sequences_chunk(data, sequence_length, start, end):
    present_sequences = set()
    for i in range(start, end - sequence_length + 1):
        subseq = data[i:i + sequence_length]
        present_sequences.add(subseq)
    return present_sequences

def find_missing_sequences(data, sequence_length):
    data_length = len(data)
    num_workers = cpu_count()
    chunk_size = data_length // num_workers

    with Pool(processes=num_workers) as pool:
        tasks = [(data, sequence_length, i * chunk_size, data_length if i == num_workers - 1 else (i + 1) * chunk_size) for i in range(num_workers)]
        present_sequence_chunks = pool.starmap(find_missing_sequences_chunk, tasks)

    present_sequences = set().union(*present_sequence_chunks)

    all_possible_sequences = set()
    for i in range(256**sequence_length):
        sequence = bytes((i >> (8 * j)) & 0xFF for j in range(sequence_length))
        all_possible_sequences.add(sequence)

    missing_sequences = all_possible_sequences - present_sequences
    return sorted(missing_sequences)

def read_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
    except IOError as e:
        print(f"Error reading file {file_path}: {e}")
        sys.exit(1)
    return data

def write_file(file_path, data):
    try:
        with open(file_path, 'wb') as f:
            f.write(data)
    except IOError as e:
        print(f"Error writing file {file_path}: {e}")
        sys.exit(1)

def calculate_combined_size(dictionaries, data):
    total_size = 0
    for dictionary in dictionaries:
        for missing_seq, substituted_seq in dictionary.items():
            missing_len = len(missing_seq)
            substituted_len = len(substituted_seq)
            total_size += 2 + missing_len + substituted_len  # 2 bytes for lengths, plus the sequences
    total_size += 2  # Separator size (0, 255)
    total_size += len(data)
    return total_size

def write_boo_file(boo_file_path, dictionaries, data, original_extension):
    try:
        with open(boo_file_path, 'wb') as boo_file:
            # Write the original extension length and extension
            if original_extension:
                extension_length = len(original_extension)
                boo_file.write(bytes([extension_length]))
                boo_file.write(original_extension.encode())
            else:
                boo_file.write(bytes([0]))

            # Write all dictionaries to the boo file
            for dictionary in dictionaries:
                for missing_seq, substituted_seq in dictionary.items():
                    missing_seq_bytes = bytes(missing_seq)
                    substituted_seq_bytes = bytes(substituted_seq)
                    missing_len = len(missing_seq_bytes)
                    substituted_len = len(substituted_seq_bytes)
                    # Ensure lengths are within the valid range of 1 to 255
                    if missing_len > 255 or substituted_len > 255:
                        raise ValueError("Sequence length exceeds 255 bytes")
                    # Write lengths and sequences
                    boo_file.write(bytes([missing_len]))
                    boo_file.write(bytes([substituted_len]))
                    boo_file.write(missing_seq_bytes)
                    boo_file.write(substituted_seq_bytes)
            # Write the separator (0, 255)
            boo_file.write(bytes([0, 255]))
            # Write the data
            boo_file.write(data)
    except IOError as e:
        print(f"Error writing to boo file {boo_file_path}: {e}")
        sys.exit(1)

def load_dictionaries_and_data(boo_file_path):
    """
    Load the dictionaries and the compressed data from the .boo file.
    """
    dictionaries = []
    data = b""
    original_extension = ""
    with open(boo_file_path, 'rb') as f:
        # Read the original extension length and extension
        extension_length = f.read(1)[0]
        if extension_length > 0:
            original_extension = f.read(extension_length).decode()

        while True:
            missing_len_byte = f.read(1)
            if not missing_len_byte:
                break  # End of file
            missing_len = missing_len_byte[0]
            substituted_len = f.read(1)[0]
            if missing_len == 0 and substituted_len == 255:
                data = f.read()  # Separator found, read the rest as data
                break
            missing_seq = f.read(missing_len)
            substituted_seq = f.read(substituted_len)
            dictionaries.append({missing_seq: substituted_seq})
    return dictionaries, data, original_extension

def add_or_replace_extension(file_path):
    base, ext = os.path.splitext(file_path)
    return base + '.boo'

def main(file_path, total_iterations, max_length, top_n=256):
    start_time = datetime.now()
    boo_file_path = add_or_replace_extension(file_path)
    compressed = False

    if file_path.endswith('.boo'):
        # Read existing .boo file
        boo_file_path = file_path 
        dictionaries, data, original_extension = load_dictionaries_and_data(file_path)
        iterations_done = len(dictionaries)
        compressed = True
        iteration_count = iterations_done
    else:
        iteration_count = 0
        iterations_done = 0
        data = read_file(file_path)
        dictionaries = []
        base, original_extension = os.path.splitext(file_path)

    previous_combined_size = len(data)
    
    sequence_length = 1

    while sequence_length <= max_length and (total_iterations == -1 or iteration_count < total_iterations):
        original_size = len(data)
        best_iteration = 1
        now_time = datetime.now()
        timing = time_difference(now_time, start_time)
        print(f"{timing} Processing sequence length: {sequence_length}")
        missing_sequences = find_missing_sequences(data, sequence_length)
        gc.collect()
        if not missing_sequences:
            sequence_length += 1
            continue

        while missing_sequences and (total_iterations == -1 or iteration_count < total_iterations):
            subsequence_counts = extract_and_filter_subsequences(data, sequence_length, max_length)            

            gc.collect()
            
            most_common_subsequences = find_most_common_subsequences(subsequence_counts, sequence_length, top_n)
            
            gc.collect()

            if not most_common_subsequences:
                break

            highest_score_sequence, (score, occurrences) = most_common_subsequences[0]

            first_missing_sequence = missing_sequences.pop(0)

            data = data.replace(bytes(highest_score_sequence), bytes(first_missing_sequence))
            dictionary = {first_missing_sequence: highest_score_sequence}
            dictionaries.append(dictionary)

            current_combined_size = calculate_combined_size(dictionaries, data)  # Calculate the combined size

            if current_combined_size < previous_combined_size:
                best_iteration = iteration_count + 1

            gc.collect()

            previous_combined_size = min(current_combined_size, previous_combined_size)

            iteration_count += 1

            now_time = datetime.now()
            timing = time_difference(now_time, start_time)
            ratio = current_combined_size/original_size
            ratio = f"{ratio:.3f}"

            print(f"{timing} Iteration {iteration_count} for sequence length {sequence_length} completed. Size {current_combined_size}b {ratio}% Best iteration {best_iteration}")

            # Save the boo file at the end of each iteration
            write_boo_file(boo_file_path, dictionaries, data, original_extension)

        sequence_length += 1

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python ghost-compress.py <file_path> <iterations> <max_length>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    total_iterations = int(sys.argv[2])
    max_length = int(sys.argv[3])
    top_n = 256  
    main(file_path, total_iterations, max_length, top_n)
