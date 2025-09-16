Ghost Compression Algorithm

A novel compression algorithm that achieves file size reduction by identifying repeated subsequences and replacing them with unused byte patterns.

How It Works

Pattern Extraction: Analyzes input data to extract all possible subsequences and count their occurrences

Filtering: Identifies subsequences that appear multiple times in the data

Replacement Strategy: Locates unused byte patterns and uses them to replace the most frequently occurring subsequences

Output: Generates compressed files with reduced size while maintaining data integrity

Installation

git clone https://github.com/RAZZULLIX/ghost.git

cd ghost

Ensure you have Python and required dependencies installed.

Usage

python ghost-compress.py <file_path> <iterations> <max_length>

Parameters:

file_path: Path to the input file (supports both regular files and previously compressed .boo files)

iterations: Number of compression cycles to perform (use -1 for infinite iterations, but you'll have to manually stop it)

max_length: Maximum length of subsequences to analyze (higher values increase memory usage)

Example:

python ghost-compress.py myfile.txt 100 10

Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request.

License

This project is licensed under GPL 3.0.
