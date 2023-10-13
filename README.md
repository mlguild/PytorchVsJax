# Parallel Model Benchmarking: PyTorch vs Jax

This repository aims to compare the performance of PyTorch and Jax when executing forward and backward passes across multiple models in parallel. The focus is on utilizing batch operations for both convolutional and multi-layer perceptron (MLP) models.

## Overview

Modern deep learning frameworks like PyTorch and Jax provide powerful tools to build, train, and deploy neural networks. One area of interest is understanding their performance characteristics when processing multiple models in parallel. This is particularly relevant in scenarios where multiple neural network models need to be evaluated simultaneously, such as in ensemble methods or hyperparameter tuning.

This benchmarking exercise revolves around:
1. Generating random input tensors of shape NxBxCxHxW.
2. Fprop and bprop operations on Nx MLPs.
3. The same operations on convolutional layers.
4. Benchmarking a simple 4-layer convolutional network and a 4-layer MLP.

## Directory Structure

├── batch_ops_example.py # Example usage of batch operations
├── batch_ops.py # Core implementation of batch operations
├── bench_pytorch.py # PyTorch benchmarking script
├── env.sh # Environment setup script
├── LICENSE # License information
├── pycache # Cached Python bytecode
│ └── batch_ops.cpython-311.pyc
└── README.md # This README file


## How to Run

1. Ensure you have the required libraries installed. This primarily includes PyTorch and its dependencies.
2. Source the environment setup script: `source env.sh`.
3. Navigate to the repository root and run `python bench_pytorch.py`. This will execute the PyTorch benchmarks and log the results.

## Observations

Upon running the benchmarks, the system will log the mean, standard deviation, maximum, and minimum timings for each operation, providing insights into the performance characteristics of PyTorch for these operations.

## Future Work

Similar benchmarks for Jax will be added to provide a direct comparison between the two frameworks.

## Contributions

Feel free to raise issues or pull requests if you have suggestions or find any bugs in the benchmarks.

## License

Please refer to the `LICENSE` file for licensing details.
