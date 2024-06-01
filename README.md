# Ghost: The Spooktacular Compression Algorithm 👻

Welcome to Ghost, the eeriest compression algorithm you'll ever encounter! Ghost works its spectral magic by finding sequences that aren't there and using them to substitute larger sequences. It's a phantom in the binary night, reducing file sizes with spooky efficiency.

## How It Works 🎃

- **Extraction:** Ghost prowls your data, extracting all possible subsequences and counting their occurrences.
- **Filtering:** Only the subsequences that occur more than once survive the spectral filter.
- **Finding the Missing:** Ghost hunts for sequences that are missing from the data, lurking in the shadows of potential patterns.
- **Replacement:** The most common sequences are replaced by these ghostly missing sequences, creating compressed data that fits in the haunted .boo extension.

## The .boo Extension 👻

When Ghost haunts your files, it compresses them into a .boo file, carrying the essence of your original data in a spooky, compact form. Don't be afraid to open it—it's just smaller!

## Installation 🕸️

Clone this repository to your local lair:

    git clone https://github.com/RAZZULLIX/ghost.git
    cd ghost

Make sure you have Python and all required packages installed, or Ghost might get cranky.

## Usage 🦇

Run Ghost from the command line with:

    python ghost-compress.py <file_path> <iterations> <max_length>

- **file_path:** Path to the file you wish to compress. You can also further compress a .boo file.
- **iterations:** Number of iterations for the ghostly compression cycle. Use -1 for endless haunting.
- **max_length:** The maximum length of sequences Ghost should consider in its eerie evaluations. (Beware of high numbers or your memory will be cursed!)

Example:

    python ghost-compress.py myfile.txt 100 10

## Contributing 👻

Feel the chill of inspiration? Fork this repo, create a new branch, and raise a PR. Let’s make Ghost even spookier together!

## License 🧛

This project is open-sourced under GPL 3.0.

## Acknowledgments 🙏

- God for the ideas
- Sonic (custom GPT-4 instance) for the code
- The open-source community for continuous inspiration and collaboration
