# Acc-SpMM


- This repository is the official implementation of Acc-SpMM: Accelerating General-purpose Sparse Matrix-Matrix Multiplication with GPU Tensor Cores, PPoPP2025.
    - [arXiv](https://arxiv.org/pdf/2501.09251)
    - [ACM Digital Library](https://dl.acm.org/doi/10.1145/3710848.3710888)


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

## Citation

To cite this project, you can use the following BibTex citation.

```
@inproceedings{zhao2025acc,
  title={Acc-SpMM: Accelerating General-purpose Sparse Matrix-Matrix Multiplication with GPU Tensor Cores},
  author={Zhao, Haisha and Li, San and Wang, Jiaheng and Zhou, Chunbao and Wang, Jue and Xin, Zhikuang and Li, Shunde and Liang, Zhiqiang and Pan, Zhijie and Liu, Fang and others},
  booktitle={Proceedings of the 30th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming},
  pages={326--338},
  year={2025}
}
```
