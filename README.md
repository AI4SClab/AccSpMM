
## Requirements

- Supported GPU: 
    - A800 
    - H100
    - RTX4090
- CUDA >= 11.8

## Usage

1. Clone the repository

    ```
    git clone https://github.com/AI4SClab/AccSpMM.git
    cd AccSpMM
    ```

2. Run the code

    ```
    mkdir build && cd build
    cmake .. && make
    cd ..
    ./mma $file_path $feature_dim   # you can use the matrix in folder dataset/test
    ```

