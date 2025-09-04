import sys
import os
import gc
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
    if sequence_length > 3:
        print(f"Warning: Generating all possible sequences for length {sequence_length} is computationally expensive.")
    try:
        all_possible_sequences = {bytes((i >> (8 * j)) & 0xFF for j in range(sequence_length)) for i in range(256 ** sequence_length)}
    except MemoryError:
        print(f"Error: Ran out of memory trying to generate 256^{sequence_length} possible sequences. Aborting.")
        return []
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
    if not data_tensor.is_cuda:
        data_tensor = data_tensor.to('cuda')

    best_overall_sequences = []
    best_overall_scores = []
    min_length = missing_sequence_length + 1
    
    powers = (256**torch.arange(8, device='cuda', dtype=torch.int64))

    for length in range(min_length, max_length + 1):
        if len(data_tensor) < length:
            continue

        num_sequences = len(data_tensor) // length
        if num_sequences == 0:
            continue
        sequences = data_tensor[:num_sequences * length].view(num_sequences, length)

        if length <= 8:
            packed_sequences = (sequences.long() * powers[:length]).sum(dim=1)
            unique_packed, counts = torch.unique(packed_sequences, return_counts=True)
            
            mask = counts > 1
            if not mask.any(): continue
            
            unique_packed = unique_packed[mask]
            counts = counts[mask]

            unpacked_sequences = (unique_packed.unsqueeze(1) // powers[:length]) % 256
            unique_sequences = unpacked_sequences.byte()
        else:
            unique_sequences, counts = torch.unique(sequences, return_counts=True, dim=0)
            mask = counts > 1
            if not mask.any(): continue
            unique_sequences = unique_sequences[mask]
            counts = counts[mask]
        
        scores = counts * (length - missing_sequence_length)
        
        num_candidates = min(top_n, len(scores))
        if num_candidates == 0: continue
            
        top_indices = torch.topk(scores, num_candidates).indices
        
        best_sequences_for_length = unique_sequences[top_indices].cpu().tolist()
        best_scores_for_length = scores[top_indices].cpu().tolist()
        best_counts_for_length = counts[top_indices].cpu().tolist()
        
        best_overall_sequences.extend(best_sequences_for_length)
        best_overall_scores.extend(zip(best_scores_for_length, best_counts_for_length))

    if not best_overall_scores:
        return [], []

    combined = list(zip(best_overall_sequences, best_overall_scores))
    combined.sort(key=lambda x: x[1][0], reverse=True)
    
    top_combined = combined[:top_n]
    
    if not top_combined: return [], []
        
    top_sequences, top_scores = zip(*top_combined)
    
    return list(top_sequences), list(top_scores)

def calculate_boo_size(dictionaries, data, original_extension):
    boo_size = 1 + len(original_extension.encode())
    for dictionary in dictionaries:
        for missing_seq, substituted_seq in dictionary.items():
            boo_size += 2 + len(missing_seq) + len(substituted_seq)
    boo_size += 2
    boo_size += len(data)
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
        print(f"{get_timestamp()} Processing sequence length: {sequence_length}", end='...')
        missing_sequences = find_missing_sequences(data, sequence_length)
        if not missing_sequences:
            sequence_length += 1
            print(" No missing sequences found.")
            continue
        else:
            print(f" Found {len(missing_sequences)} sequences")

        while missing_sequences and (total_iterations == -1 or iteration_count < total_iterations):
            data_tensor = torch.tensor(np.frombuffer(data, dtype=np.uint8), device='cuda')
            top_sequences, scores = find_top_n_sequences_cuda(data_tensor, sequence_length, max_length, top_n)
            
            del data_tensor
            gc.collect()
            torch.cuda.empty_cache()

            if not top_sequences:
                break
            
            substitutions_in_this_pass = 0
            used_bytes = set()
            for highest_score_sequence in top_sequences:
                if total_iterations != -1 and iteration_count >= total_iterations:
                    break

                highest_score_sequence_bytes = bytes(highest_score_sequence)

                if any(byte in used_bytes for byte in highest_score_sequence_bytes):
                    break # YOUR CORRECT LOGIC: Stop the entire pass on conflict

                if not missing_sequences:
                    break
                
                first_missing_sequence = bytes(missing_sequences.pop(0))
                data_before_replace = data
                
                data = data.replace(highest_score_sequence_bytes, first_missing_sequence)
                
                data_after_reverse_replace = data.replace(first_missing_sequence, highest_score_sequence_bytes)
                
                if data_before_replace == data_after_reverse_replace:
                    used_bytes.update(highest_score_sequence_bytes)
                    substitutions_in_this_pass += 1
                    iteration_count += 1
                    
                    dictionaries.append({first_missing_sequence: highest_score_sequence_bytes})

                    if iteration_count % 100 == 0:
                        write_boo_file(boo_file_path, dictionaries, data, original_extension)

                    new_size = calculate_boo_size(dictionaries, data, original_extension)
                    ratio = f"{(new_size / original_size) * 100:.3f}%"
                    print(f"{get_timestamp()} Iteration {iteration_count} for sequence length {sequence_length} completed. Size {new_size}b Ratio {ratio} Substituted sequence length {len(highest_score_sequence_bytes)}")
                else:
                    data = data_before_replace

            if substitutions_in_this_pass == 0:
                print(f"{get_timestamp()} No more valid replacements found for this data state. Moving to next length.")
                break
        
        sequence_length += 1
    
    write_boo_file(boo_file_path, dictionaries, data, original_extension)
    print(f"{get_timestamp()} Compression finished.")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python ghost-compress-cuda.py <file_path> <iterations> <max_length>")
        sys.exit(1)

    file_path = sys.argv[1]
    total_iterations = int(sys.argv[2])
    max_length = int(sys.argv[3])
    
    if max_length > 255:
        max_length = 255
    
    if not torch.cuda.is_available():
        print("CUDA is not available. This script requires a CUDA-enabled GPU.")
        sys.exit(1)

    main(file_path, total_iterations, max_length)
