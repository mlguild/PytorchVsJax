#!/bin/bash

declare -a dtypes=("float32" "float16" "bfloat16")
declare -a batch_sizes=("1" "10" "50" "100" "500")
declare -a num_models=("1" "10" "50" "100")
declare -a bench_type=("single_mlp", "single_conv", "four_mlp", "four_conv")

for dtype in "${dtypes[@]}"; do
  for B in "${num_models[@]}"; do
    for N in "${batch_sizes[@]}"; do
        for bench in "${bench_type[@]}"; do
            echo "Running jax/bench.py with B=$B, N=$N, dtype=$dtype, benchmark_type=$bench"
            python jax/bench.py --B="$B" --N="$N" --dtype="$dtype" --benchmark_type="$bench"
            sleep 5
        done
    done
  done
done
