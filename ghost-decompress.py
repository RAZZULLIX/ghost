import sys
import pickle

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

def load_dictionaries(dict_file_path):
    """
    Load the dictionaries from a file.
    """
    mappings = []
    with open(dict_file_path, 'rb') as f:
        try:
            while True:
                mappings.append(pickle.load(f))
        except EOFError:
            pass
    return mappings

def decompress(data, mappings):
    """
    Decompress the data using the mappings.
    Apply the mappings in reverse order.
    """
    for key, value in reversed(mappings):
        data = data.replace(bytes(key), bytes(value))
    return data

def main(file_path):
    # Read the compressed file
    compressed_data = read_file(file_path)

    # Load the dictionaries
    dict_file_path = file_path.replace('.boo', '.dict')
    mappings = load_dictionaries(dict_file_path)

    # Decompress the data
    decompressed_data = decompress(compressed_data, mappings)

    # Save the decompressed file
    decompressed_file_path = file_path.replace('.boo', '')
    decompressed_file_path = 'deco_' + decompressed_file_path
    write_file(decompressed_file_path, decompressed_data)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ghost-decompress.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    main(file_path)
