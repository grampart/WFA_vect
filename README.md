# WFA Vectorization Implementation

This repository contains code that extends the WFA (Wavefront Alignment) comparison process with vectorized implementations using AVX2 and AVX512. 
The rest of the code is sourced from [https://github.com/smarco/WFA-paper](https://github.com/smarco/WFA-paper). The usage of this repository is consistent with the instructions provided in the original repository.

## Usage

To use this implementation, follow the same steps as outlined in [https://github.com/smarco/WFA-paper](https://github.com/smarco/WFA-paper). 

### Quick Start

1. In the root directory, run:
   ```bash
   make clean all
   ```
2. The code includes a usage example located in the `./example` directory. To run the example, navigate to the directory and execute:
   ```bash
   bash run.sh
   ```

## Important Note

This code does not check whether the system supports AVX instructions. If your system does not support AVX, the compilation will result in errors.
