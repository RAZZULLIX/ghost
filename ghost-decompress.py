import sys
from datetime import datetime

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

def write_file(file_path, data):
    with open(file_path, 'wb') as f:
        f.write(data)

def load_dictionaries_and_data(boo_file_path):
    with open(boo_file_path, 'rb') as f:
        dictionaries = []

        # Read the original extension length and extension
        extension_length = f.read(1)[0]
        if extension_length > 0:
            original_extension = f.read(extension_length).decode()
        else:
            original_extension = ""

        while True:
            missing_len_byte = f.read(1)
            if not missing_len_byte:
                break  # End of file
            missing_len = missing_len_byte[0]
            substituted_len = f.read(1)[0]
            if missing_len == 0 and substituted_len == 255:
                break  # Separator found
            missing_seq = f.read(missing_len)
            substituted_seq = f.read(substituted_len)
            dictionaries.append((missing_seq, substituted_seq))
        data = f.read()
    return dictionaries, data, original_extension

def decompress(data, dictionaries):
    i = 1
    for missing_seq, substituted_seq in reversed(dictionaries):        
        print(f"\r{get_timestamp()} Processing dictionary {i}/{len(dictionaries)}...", end="", flush=True)
        data = data.replace(missing_seq, substituted_seq)
        i = i + 1
    return data

def main(file_path):
    # Read the .boo file
    print(f"{get_timestamp()} Beginning decompression...")
    dictionaries, compressed_data, original_extension = load_dictionaries_and_data(file_path)

    # Decompress the data
    decompressed_data = decompress(compressed_data, dictionaries)

    # Save the decompressed file
    decompressed_file_path = file_path.replace('.boo', original_extension)
    decompressed_file_path = 'deco_' + decompressed_file_path
    write_file(decompressed_file_path, decompressed_data)
    print(f"{get_timestamp()} Decompression completed successfully!!!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ghost-decompress.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    main(file_path)
