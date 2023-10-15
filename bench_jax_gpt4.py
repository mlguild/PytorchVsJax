import time
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
from jax import random

import wandb


def benchmark(
    apply_fn: nn.Module, params: dict, x: jnp.ndarray, R: int = 1000
) -> Dict[str, float]:
    timings = []

    for _ in range(R):
        start_time = time.time()
        out = apply_fn(params, x)
        end_time = time.time()
        timings.append(end_time - start_time)

    mean_time = sum(timings) / R
    std_time = (sum([(mean_time - t) ** 2 for t in timings]) / R) ** 0.5
    max_time = max(timings)
    min_time = min(timings)
    median_time = sorted(timings)[R // 2]

    return {
        "mean": mean_time,
        "std": std_time,
        "max": max_time,
        "min": min_time,
        "median": median_time,
    }


class ConvNetWithGlobalPooling(nn.Module):
    channel_in: int

    def setup(self):
        self.conv1 = nn.Conv(features=64, kernel_size=(3, 3))
        self.conv2 = nn.Conv(features=128, kernel_size=(3, 3))
        self.conv3 = nn.Conv(features=256, kernel_size=(3, 3))
        self.fc = nn.Dense(features=10)

    def __call__(self, x):
        x = nn.relu(self.conv1(x))
        x = nn.relu(self.conv2(x))
        x = nn.relu(self.conv3(x))
        x = jnp.mean(x, axis=(2, 3))  # global average pooling
        x = x.reshape((x.shape[0], -1))
        x = self.fc(x)
        return x


class FourLayerMLP(nn.Module):
    def setup(self):
        self.fc1 = nn.Dense(features=512)
        self.fc2 = nn.Dense(features=256)
        self.fc3 = nn.Dense(features=128)
        self.fc4 = nn.Dense(features=10)

    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        x = nn.relu(self.fc1(x))
        x = nn.relu(self.fc2(x))
        x = nn.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def main():
    key = random.PRNGKey(0)
    C, H, W = 3, 32, 32
    num_classes = 10

    for B in [1, 10, 50, 100]:
        for N in [1, 10, 50, 100, 500]:
            wandb.init(
                project="jax-vs-pytorch-benchmarking",
                name=f"jax-benchmark-B{B}-N{N}",
                reinit=True,
            )
            wandb.config.num_models = B
            wandb.config.batch_size = N

            x = random.normal(key, (B, N, C, H, W))
            y = random.randint(key, (B, N), 0, num_classes)

            # ConvNet Benchmark
            model = ConvNetWithGlobalPooling(channel_in=C)
            params = model.init(key, x)["params"]
            conv_net_time = benchmark(model.apply, params, x)
            wandb.log(
                {"ConvNet with Global Pooling Benchmark Time": conv_net_time}
            )

            # 4-layer MLP Benchmark
            model = FourLayerMLP()
            params = model.init(key, x)["params"]
            mlp_net_time = benchmark(model.apply, params, x)
            wandb.log({"4-layer MLP Benchmark Time": mlp_net_time})

            wandb.log({"num_models": B, "batch_size": N})


if __name__ == "__main__":
    main()
