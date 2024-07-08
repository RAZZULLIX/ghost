import sys

def read_file(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()
    return data

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
    for missing_seq, substituted_seq in reversed(dictionaries):
        data = data.replace(missing_seq, substituted_seq)
    return data

def main(file_path):
    # Read the .boo file
    dictionaries, compressed_data, original_extension = load_dictionaries_and_data(file_path)

    # Decompress the data
    decompressed_data = decompress(compressed_data, dictionaries)

    # Save the decompressed file
    decompressed_file_path = file_path.replace('.boo', original_extension)
    decompressed_file_path = 'deco_' + decompressed_file_path
    write_file(decompressed_file_path, decompressed_data)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ghost-decompress.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    main(file_path)
