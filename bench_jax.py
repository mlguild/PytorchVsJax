from time import time
from functools import partial
from typing import Any, Sequence

import torch
import jax
from flax import linen as nn
import wandb
import jax.numpy as jnp
from jax import config
config.update("jax_disable_jit", True)

class MLP(nn.Module):
    hidden_dims : Sequence[int]
    num_classes : int
    dtype : jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        for dims in self.hidden_dims:
            x = nn.relu(nn.Dense(dims, dtype=self.dtype)(x))
        return nn.Dense(num_classes, dtype=self.dtype)(x)

class CNN(nn.Module):
    out_filters : Sequence[int]
    kernel_sizes : Sequence[tuple]
    num_classes : int
    dtype : jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x) -> Any:
        for out_filter, kernel_size in zip(self.out_filters, self.kernel_sizes):
            x = nn.relu(nn.Conv(features=out_filter, kernel_size=kernel_size, dtype=self.dtype)(x))
        # the x dimension here will be (batch_size, H, W, out_filter)
        # after the pooling operation, it will be (batch_size, out_filter)
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
        return x

class Trainer:
    def __init__(self, model_class, num_devices, model_hparams) -> None:
        self.model = model_class(**model_hparams)
        self.num_devices = num_devices

    def init_model(self, rng, x):
        subkeys = jax.random.split(rng, self.num_devices)
        params = jax.vmap(self.model.init, in_axes=(0, 0))(subkeys, x)
        return params

    @partial(jax.jit, static_argnums=(0,))
    def loss(self, params, x, y):
        logits = self.model.apply(params, x)
        one_hot_labels = jax.nn.one_hot(y, num_classes)
        loss = jnp.mean(jax.nn.log_softmax(logits) * one_hot_labels)
        return -loss
    
    def benchmark(self, params, x, y = None, backward = True, R=100):
        """
        Benchmark a given layer/model for forward and backward pass.
        """
        fprop_timings = []
        bprop_timings = []
        combined_timings = []
        
        forward = jax.vmap(jax.jit(self.model.apply), in_axes=(0, 0))
        vmapped_val_and_grad = jax.vmap(jax.value_and_grad(self.loss), in_axes=(0, 0, 0))
        
        for _ in range(R):
            start_time = time()
            out = forward(params, x)
            fprop_timings.append(time() - start_time)

            if y is not None and backward:
                start_time = time()
                out, _ = vmapped_val_and_grad(params, x, y)
                end_time = time()
                bprop_timings.append(time() - start_time - fprop_timings[-1])

            combined_timings.append(
                fprop_timings[-1] + (bprop_timings[-1] if bprop_timings else 0)
                )
            
        def compute_stats(timings):
                timings_tensor = torch.tensor(timings)
                mean_time = torch.mean(timings_tensor).item()
                std_time = torch.std(timings_tensor).item()
                max_time = torch.max(timings_tensor).item()
                min_time = torch.min(timings_tensor).item()
                median_time = torch.median(timings_tensor).item()

                return {
                    "mean": mean_time,
                    "std": std_time,
                    "max": max_time,
                    "min": min_time,
                    "median": median_time,
                }

        return {
            "fprop": compute_stats(fprop_timings),
            "bprop": compute_stats(bprop_timings),
            "combined": compute_stats(combined_timings),
        }

if __name__ == '__main__':
    C, H, W = 3, 32, 32
    num_classes = 10
    dtypes = [jnp.float32, jnp.float16, jnp.bfloat16]

    key = jax.random.PRNGKey(0)

    for dtype in dtypes:
        for B in [1, 10, 50, 100]:
            for N in [1, 10, 50, 100, 500]:
                try:
                    wandb.init(
                                project="pytorch-vs-jax-benchmarking",
                                name=f"jax-benchmark-B{B}-N{N}-dtype{dtype}",
                                reinit=True,
                            )
                    wandb.config.num_models = B
                    wandb.config.batch_size = N

                    x = jax.random.normal(key, shape=(B, N, H, W, C), dtype=dtype)
                    y = jax.random.randint(key, (B,N), 0, num_classes)

                    # Single Layer MLP
                    mlp_args = {"num_classes": num_classes,
                                "hidden_dims": [],
                                "dtype": dtype
                                }
                    trainer = Trainer(MLP, B, mlp_args)
                    params = trainer.init_model(key, x)
                    timings = trainer.benchmark(params, x, backward=False)
                    wandb.log({"MLP Benchmark Time": timings})

                    # Single Layer Convolution
                    cnn_args = {"features": num_classes,
                                "kernel_size": (3, 3),
                                "dtype": dtype}
                    trainer = Trainer(nn.Conv, B, cnn_args)
                    params = trainer.init_model(key, x)
                    timings = trainer.benchmark(params, x, backward=False)
                    wandb.log({"Conv Layer Benchmark Time": timings})

                    # 4-Layer MLP Layer Benchmark
                    mlp_args = {"num_classes": num_classes,
                                "hidden_dims": [512, 256, 128],
                                "dtype": dtype
                                }
                    trainer = Trainer(MLP, B, mlp_args)
                    params = trainer.init_model(key, x)
                    timings = trainer.benchmark(params, x, y=y)
                    wandb.log({"4-layer MLP Benchmark Time": timings})

                    # CNN
                    cnn_args = {"out_filters": [64, 128, 256],
                                "kernel_sizes": [(3, 3), (3, 3), (3, 3)],
                                "num_classes": num_classes,
                                "dtype": dtype
                                }
                    trainer = Trainer(CNN, B, cnn_args)
                    params = trainer.init_model(key, x)
                    timings = trainer.benchmark(params, x, y=y)
                    wandb.log(
                                {
                                    "ConvNet with Global Pooling Benchmark Time": timings
                                }
                            )
                    
                    wandb.log({"num_models": B, "batch_size": N})
                except Exception as e:
                    # Log the exception to the console
                    print(f"Error during benchmark with B={B} and N={N}: {e}")
                    # Log the exception to wandb
                    wandb.log({"error": str(e)})
