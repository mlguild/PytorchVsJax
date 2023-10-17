import os
from functools import partial
from time import time
from typing import Any, Sequence

import fire
import jax
import jax.numpy as jnp
from flax import linen as nn

import wandb

# Get environment variables
WANDB_ENTITY = os.getenv('WANDB_ENTITY')
WANDB_PROJECT = os.environ.get('WANDB_PROJECT')

class MLP(nn.Module):
    hidden_dims: Sequence[int]
    num_classes: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        for dims in self.hidden_dims:
            x = nn.relu(nn.Dense(dims, dtype=self.dtype)(x))
        return nn.Dense(self.num_classes, dtype=self.dtype)(x)


class CNN(nn.Module):
    out_filters: Sequence[int]
    kernel_sizes: Sequence[tuple]
    num_classes: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x) -> Any:
        for out_filter, kernel_size in zip(
            self.out_filters, self.kernel_sizes
        ):
            x = nn.relu(
                nn.Conv(
                    features=out_filter,
                    kernel_size=kernel_size,
                    dtype=self.dtype,
                )(x)
            )
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
        return x


class Trainer:
    def __init__(
        self, model_class, num_devices, model_hparams, num_classes
    ) -> None:
        self.model = model_class(**model_hparams)
        self.num_devices = num_devices
        self.num_classes = num_classes

    def init_model(self, rng, x):
        subkeys = jax.random.split(rng, self.num_devices)
        params = jax.vmap(self.model.init, in_axes=(0, 0))(subkeys, x)
        return params

    @partial(jax.jit, static_argnums=(0,))
    def loss(self, params, x, y):
        logits = self.model.apply(params, x)
        one_hot_labels = jax.nn.one_hot(y, self.num_classes)
        loss = jnp.mean(jax.nn.log_softmax(logits) * one_hot_labels)
        return -loss

    @partial(jax.jit, static_argnums=(0,))
    def forward_pass(self, params, x):
        return self.model.apply(params, x)

    def benchmark(self, params, x, y=None, backward=True, R=100):
        fprop_timings = []
        bprop_timings = []
        combined_timings = []

        forward = jax.vmap(self.forward_pass, in_axes=(0, 0))
        vmapped_val_and_grad = jax.vmap(
            jax.value_and_grad(self.loss), in_axes=(0, 0, 0)
        )

        # Run forward and forward/backward passes once to compile the routines
        forward(params, x)
        vmapped_val_and_grad(params, x, y)

        for _ in range(R):
            start_time = time()
            out = forward(params, x)
            fprop_timings.append(time() - start_time)

            if y is not None and backward:
                start_time = time()
                out, _ = vmapped_val_and_grad(params, x, y)
                bprop_timings.append(time() - start_time - fprop_timings[-1])

            combined_timings.append(
                fprop_timings[-1] + (bprop_timings[-1] if bprop_timings else 0)
            )

        def compute_stats(timings):
            return {
                "mean": sum(timings) / len(timings),
                "std": (
                    sum(
                        (x - sum(timings) / len(timings)) ** 2 for x in timings
                    )
                    / len(timings)
                )
                ** 0.5,
                "max": max(timings),
                "min": min(timings),
                "median": sorted(timings)[len(timings) // 2],
            }

        return {
            "fprop": compute_stats(fprop_timings),
            "bprop": compute_stats(bprop_timings) if backward else None,
            "combined": compute_stats(combined_timings),
        }


def run_benchmark(B=1, N=1, dtype="float32", benchmark_type="single_mlp"):
    dtype = getattr(jnp, dtype)  # Map string to actual JAX dtype

    C, H, W = 3, 32, 32
    num_classes = 10
    key = jax.random.PRNGKey(0)

    x = jax.random.normal(key, shape=(B, N, H, W, C), dtype=dtype)
    y = jax.random.randint(key, (B, N), 0, num_classes)
    mode = "jit"

    wandb.init(
        project="pytorch-vs-jax-benchmarking",
        name=f"jax-benchmark-B{B}-N{N}-dtype{dtype}-mode{mode}-type{benchmark_type}",
        reinit=False,
    )
    wandb.config.num_models = B
    wandb.config.batch_size = N
    wandb.config.dtype = dtype
    wandb.config.framework = "jax"

    if benchmark_type == "single_mlp":
        # Single Layer MLP
        mlp_args = {
            "num_classes": num_classes,
            "hidden_dims": [],
            "dtype": dtype,
        }
        trainer = Trainer(MLP, B, mlp_args, num_classes=num_classes)
        params = trainer.init_model(key, x)
        timings = trainer.benchmark(params, x, y, backward=True)
        wandb.log({"MLP Benchmark Time": timings})

    elif benchmark_type == "single_conv":
        # Single Layer Convolution
        cnn_args = {
            "out_filters": [num_classes],
            "kernel_sizes": [(3, 3)],
            "num_classes": num_classes,
            "dtype": dtype,
        }
        trainer = Trainer(CNN, B, cnn_args, num_classes=num_classes)
        params = trainer.init_model(key, x)
        timings = trainer.benchmark(params, x, backward=False)
        wandb.log({"Conv Layer Benchmark Time": timings})

    elif benchmark_type == "four_mlp":
        # 4-Layer MLP
        mlp_args = {
            "num_classes": num_classes,
            "hidden_dims": [256, 256, 256],
            "dtype": dtype,
        }
        trainer = Trainer(MLP, B, mlp_args, num_classes=num_classes)
        params = trainer.init_model(key, x)
        timings = trainer.benchmark(params, x, y, backward=True)
        wandb.log({"4-layer MLP Benchmark Time": timings})
    elif benchmark_type == "four_conv":
        # 4-Layer Convolution
        cnn_args = {
            "out_filters": [32, 64, 128, 256],
            "kernel_sizes": [(3, 3), (3, 3), (3, 3), (3, 3)],
            "num_classes": num_classes,
            "dtype": dtype,
        }
        trainer = Trainer(
            CNN,
            B,
            cnn_args,
            num_classes=num_classes,
        )
        params = trainer.init_model(key, x)
        timings = trainer.benchmark(params, x, y, backward=True)
        wandb.log({"ConvNet with Global Pooling Benchmark Time": timings})


if __name__ == "__main__":
    fire.Fire(run_benchmark)
