import sys
import os
import gc
from collections import Counter
from multiprocessing import Pool, cpu_count
from datetime import datetime

def time_difference(start_time, end_time):
    diff = abs(end_time - start_time)
    total_seconds = int(diff.total_seconds())
    milliseconds = int((diff.total_seconds() - total_seconds) * 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"[{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}]" if hours <= 99 else "[99:59:59.999]+"

def merge_counters(counters):
    result = Counter()
    for counter in counters:
        result.update(counter)
    return result

def filter_subsequences(subsequences):
    return {k: v for k, v in subsequences.items() if v > 1}

def extract_and_filter_subsequences(data, min_length=1, max_length=256):
    data_length = len(data)
    num_workers = max(int(cpu_count() // 1.5), 1)
    chunk_size = (data_length + num_workers - 1) // num_workers

    with Pool(processes=num_workers) as pool:
        tasks = [(data, min_length, max_length, i * chunk_size, min((i + 1) * chunk_size, data_length)) for i in range(num_workers)]
        subsequence_chunks = pool.starmap(extract_subsequences_chunk, tasks)

    subsequences = merge_counters(subsequence_chunks)
    return filter_subsequences(subsequences)

def extract_subsequences_chunk(data, min_length, max_length, start, end):
    subsequences = Counter()
    for length in range(min_length, max_length + 1):
        for i in range(start, end - length + 1):
            subseq = data[i:i + length]
            subsequences[subseq] += 1
    return subsequences

def find_most_common_subsequences(subsequence_counts, missing_sequence_length, top_n=256):
    scored_subsequences = {k: ((len(k) - missing_sequence_length) * v, v) for k, v in subsequence_counts.items()}
    return sorted(scored_subsequences.items(), key=lambda item: item[1][0], reverse=True)[:top_n]

def find_missing_sequences_chunk(data, sequence_length, start, end):
    return {data[i:i + sequence_length] for i in range(start, end - sequence_length + 1)}

def find_missing_sequences(data, sequence_length):
    data_length = len(data)
    num_workers = max(int(cpu_count() // 1.5), 1)
    chunk_size = (data_length + num_workers - 1) // num_workers

    with Pool(processes=num_workers) as pool:
        tasks = [(data, sequence_length, i * chunk_size, min((i + 1) * chunk_size, data_length)) for i in range(num_workers)]
        present_sequence_chunks = pool.starmap(find_missing_sequences_chunk, tasks)

    present_sequences = set().union(*present_sequence_chunks)
    all_possible_sequences = {bytes((i >> (8 * j)) & 0xFF for j in range(sequence_length)) for i in range(256 ** sequence_length)}
    return sorted(all_possible_sequences - present_sequences)

def read_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            return f.read()
    except IOError as e:
        print(f"Error reading file {file_path}: {e}")
        sys.exit(1)

def write_file(file_path, data):
    try:
        with open(file_path, 'wb') as f:
            f.write(data)
    except IOError as e:
        print(f"Error writing file {file_path}: {e}")
        sys.exit(1)

def write_boo_file(boo_file_path, dictionaries, data, original_extension):
    try:
        with open(boo_file_path, 'wb') as boo_file:
            boo_file.write(bytes([len(original_extension)]))
            boo_file.write(original_extension.encode())

            for dictionary in dictionaries:
                for missing_seq, substituted_seq in dictionary.items():
                    boo_file.write(bytes([len(missing_seq)]))
                    boo_file.write(bytes([len(substituted_seq)]))
                    boo_file.write(missing_seq)
                    boo_file.write(substituted_seq)
            boo_file.write(bytes([0, 255]))
            boo_file.write(data)

        return os.path.getsize(boo_file_path)
    except IOError as e:
        print(f"Error writing to boo file {boo_file_path}: {e}")
        sys.exit(1)

def load_dictionaries_and_data(boo_file_path):
    dictionaries = []
    with open(boo_file_path, 'rb') as f:
        extension_length = f.read(1)[0]
        original_extension = f.read(extension_length).decode() if extension_length > 0 else ""
        while True:
            missing_len_byte = f.read(1)
            if not missing_len_byte:
                break
            missing_len = missing_len_byte[0]
            substituted_len = f.read(1)[0]
            if missing_len == 0 and substituted_len == 255:
                data = f.read()
                break
            missing_seq = f.read(missing_len)
            substituted_seq = f.read(substituted_len)
            dictionaries.append({missing_seq: substituted_seq})
    return dictionaries, data, original_extension

def add_or_replace_extension(file_path):
    return os.path.splitext(file_path)[0] + '.boo'

def main(file_path, total_iterations, max_length, top_n=256):
    start_time = datetime.now()
    boo_file_path = add_or_replace_extension(file_path)

    if file_path.endswith('.boo'):
        boo_file_path = file_path
        dictionaries, data, original_extension = load_dictionaries_and_data(file_path)
        iteration_count = len(dictionaries)
    else:
        data = read_file(file_path)
        dictionaries = []
        original_extension = os.path.splitext(file_path)[1]
        iteration_count = 0

    original_size = os.path.getsize(file_path)
    sequence_length = 1

    while sequence_length <= max_length and (total_iterations == -1 or iteration_count < total_iterations):
        now_time = datetime.now()
        timing = time_difference(now_time, start_time)
        print(f"{timing} Processing sequence length: {sequence_length}")
        missing_sequences = find_missing_sequences(data, sequence_length)
        if not missing_sequences:
            sequence_length += 1
            continue

        while missing_sequences and (total_iterations == -1 or iteration_count < total_iterations):
            subsequence_counts = extract_and_filter_subsequences(data, sequence_length, max_length)
            most_common_subsequences = find_most_common_subsequences(subsequence_counts, sequence_length, top_n)
            if not most_common_subsequences:
                break

            highest_score_sequence, (score, occurrences) = most_common_subsequences[0]
            first_missing_sequence = missing_sequences.pop(0)

            data = data.replace(bytes(highest_score_sequence), bytes(first_missing_sequence))
            dictionaries.append({first_missing_sequence: highest_score_sequence})
            iteration_count += 1

            now_time = datetime.now()
            timing = time_difference(now_time, start_time)

            new_size = write_boo_file(boo_file_path, dictionaries, data, original_extension)
            ratio = f"{(new_size / original_size) * 100:.3f}%"

            print(f"{timing} Iteration {iteration_count} for sequence length {sequence_length} completed. Size {new_size}b Ratio {ratio}")

        sequence_length += 1

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python ghost-compress.py <file_path> <iterations> <max_length>")
        sys.exit(1)

    file_path = sys.argv[1]
    total_iterations = int(sys.argv[2])
    max_length = int(sys.argv[3])
    main(file_path, total_iterations, max_length)
