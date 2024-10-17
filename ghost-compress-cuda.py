import sys
import os
import gc
from collections import Counter
from multiprocessing import Pool, cpu_count
from datetime import datetime
import torch
import numpy as np

# Global start time
start_time = datetime.now()

def get_timestamp():
    """Get the time difference from the start_time."""
    now = datetime.now()
    diff = abs(now - start_time)
    total_seconds = int(diff.total_seconds())
    milliseconds = int((diff.total_seconds() - total_seconds) * 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"[{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}]" if hours <= 99 else "[99:59:59.999]+"

def find_missing_sequences_chunk(data, sequence_length, start, end):
    return {data[i:i + sequence_length] for i in range(start, end - sequence_length + 1)}

def find_missing_sequences(data, sequence_length):
    data_length = len(data)
    num_workers = max(int(cpu_count() // 3), 1)
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
        print(f"{get_timestamp()}Error reading file {file_path}: {e}")
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
        print(f"{get_timestamp()}Error writing to boo file {boo_file_path}: {e}")
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

def find_top_n_sequences_cuda(data_tensor, missing_sequence_length, max_length, top_n=128):
    def count_sequences_cuda(data_tensor, length):
        num_sequences = (len(data_tensor) // length)
        sequences = data_tensor[:num_sequences * length].view(num_sequences, length)
        unique_sequences, counts = torch.unique(sequences, return_counts=True, dim=0)
        mask = counts > 1
        filtered_unique_sequences = unique_sequences[mask]
        filtered_counts = counts[mask]

        if len(filtered_unique_sequences) == 0:
            return None, []

        scores = filtered_counts * (length - missing_sequence_length)
        top_indices = torch.topk(scores, min(top_n, len(filtered_unique_sequences))).indices
        best_sequences = filtered_unique_sequences[top_indices]
        best_counts = filtered_counts[top_indices].cpu().tolist()
        best_scores = scores[top_indices].cpu().tolist()

        return best_sequences, list(zip(best_scores, best_counts))

    best_overall_sequences = []
    best_overall_scores = []

    for length in range(missing_sequence_length + 1, max_length + 1):
        best_sequences, scores = count_sequences_cuda(data_tensor, length)
        if best_sequences is not None:
            for i in range(len(best_sequences)):
                best_overall_sequences.append(best_sequences[i].cpu().tolist())
                best_overall_scores.append(scores[i])

    # Sort by scores and return the top N
    combined = list(zip(best_overall_sequences, best_overall_scores))
    combined.sort(key=lambda x: x[1][0], reverse=True)  # Sort by score
    top_combined = combined[:top_n]
    
    top_sequences, top_scores = zip(*top_combined) if top_combined else ([], [])
    
    return list(top_sequences), list(top_scores)

def calculate_boo_size(dictionaries, data, original_extension):
    boo_size = 1 + len(original_extension)  # For extension length and extension bytes
    for dictionary in dictionaries:
        for missing_seq, substituted_seq in dictionary.items():
            boo_size += 2 + len(missing_seq) + len(substituted_seq)  # For sequence lengths and sequences
    boo_size += 2  # For termination bytes
    boo_size += len(data)  # For data
    return boo_size

def main(file_path, total_iterations, max_length, top_n=256):
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

        print(f"{get_timestamp()} Processing sequence length: {sequence_length}", end='...'),
        missing_sequences = find_missing_sequences(data, sequence_length)
        if not missing_sequences:
            sequence_length += 1
            print()
            continue
        else:
            print(f" Found {len(missing_sequences)} sequences")

        while missing_sequences and (total_iterations == -1 or iteration_count < total_iterations):
            
            data_tensor = torch.tensor(np.frombuffer(data, dtype=np.uint8), device='cuda', dtype=torch.uint8)
            top_sequences, scores = find_top_n_sequences_cuda(data_tensor, sequence_length, max_length, top_n)
            if not top_sequences:
                break
            
            used_bytes = set()
            found_valid_replacement = False
            for highest_score_sequence in top_sequences:
                highest_score_sequence = bytearray(highest_score_sequence)
                
                if any(byte in used_bytes for byte in highest_score_sequence):
                    break
                
                while missing_sequences:
                    first_missing_sequence = missing_sequences.pop(0)
                    data1 = data
                    data = data.replace(bytes(highest_score_sequence), bytes(first_missing_sequence))
                    data2 = data.replace(bytes(first_missing_sequence), bytes(highest_score_sequence))
                    
                    if data1 == data2:
                        used_bytes.update(highest_score_sequence)
                        found_valid_replacement = True
                        break

                    else:
                        print(f"{get_timestamp()} Replacement failed for {bytes(first_missing_sequence)} with {bytes(highest_score_sequence)}")
                        data = data1

                if found_valid_replacement:
                    dictionaries.append({first_missing_sequence: highest_score_sequence})
                    iteration_count += 1

                    if iteration_count % 100 == 0:
                        new_size =  write_boo_file(boo_file_path, dictionaries, data, original_extension)

                    new_size = calculate_boo_size(dictionaries, data, original_extension)
                    ratio = f"{(new_size / original_size) * 100:.3f}%"

                    print(f"{get_timestamp()} Iteration {iteration_count} for sequence length {sequence_length} completed. Size {new_size}b Ratio {ratio} Substituted sequence length {len(highest_score_sequence)}")
                    

            if not found_valid_replacement:
                print(f"{get_timestamp()} No valid replacements found for sequence length {sequence_length}. Moving to the next length.")
                break
            
        sequence_length += 1
    
    new_size = write_boo_file(boo_file_path, dictionaries, data, original_extension)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python ghost-compress.py <file_path> <iterations> <max_length>")
        sys.exit(1)

    file_path = sys.argv[1]
    total_iterations = int(sys.argv[2])
    max_length = int(sys.argv[3])
    
    if max_length > 255:
        max_length = 255
    main(file_path, total_iterations, max_length)
