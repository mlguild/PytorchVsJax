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

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        for dims in self.hidden_dims:
            x = nn.relu(nn.Dense(dims)(x))
        return nn.Dense(num_classes)(x)

class CNN(nn.Module):
    out_filters : Sequence[int]
    kernel_sizes : Sequence[tuple]
    num_classes : int

    @nn.compact
    def __call__(self, x) -> Any:
        for out_filter, kernel_size in zip(self.out_filters, self.kernel_sizes):
            x = nn.relu(nn.Conv(features=out_filter, kernel_size=kernel_size)(x))
        # the x dimension here will be (batch_size, H, W, out_filter)
        # after the pooling operation, it will be (batch_size, out_filter)
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.num_classes)(x)
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
    
    def benchmark(self, params, x, y, R=100):
        timings = []
        vmapped_val_and_grad = jax.vmap(jax.value_and_grad(self.loss), in_axes=(0, 0, 0))
        
        for _ in range(R):
            start_time = time()
            out, _ = vmapped_val_and_grad(params, x, y)
            end_time = time()
            timings.append(end_time - start_time)
        
        mean_time = torch.tensor(timings).mean()
        std_time = torch.tensor(timings).std()
        max_time = torch.tensor(timings).max()
        min_time = torch.tensor(timings).min()
        median_time = torch.tensor(timings).median()

        return {
            "mean": mean_time,
            "std": std_time,
            "max": max_time,
            "min": min_time,
            "median": median_time,
        }

if __name__ == '__main__':
    C, H, W = 3, 32, 32
    num_classes = 10

    key = jax.random.PRNGKey(0)

    for B in [1, 10, 50, 100]:
        for N in [1, 10, 50, 100, 500]:
            try:
                x = jax.random.normal(key, shape=(B, N, H, W, C))
                y = jax.random.randint(key, (B,N), 0, num_classes)

                # Single Layer MLP
                mlp_args = {"num_classes": num_classes,
                            "hidden_dims": []
                            }
                trainer = Trainer(MLP, B, mlp_args)
                params = trainer.init_model(key, x)
                trainer.benchmark(params, x, y)

                # Single Layer Convolution
                # TODO


                # 4-Layer MLP Layer Benchmark
                mlp_args = {"num_classes": num_classes,
                            "hidden_dims": [512, 256, 128]
                            }
                trainer = Trainer(MLP, B, mlp_args)
                params = trainer.init_model(key, x)
                trainer.benchmark(params, x, y )
                
                # CNN
                cnn_args = {"out_filters": [64, 128, 256],
                            "kernel_sizes": [(3, 3), (3, 3), (3, 3)],
                            "num_classes": num_classes,
                            }
                trainer = Trainer(CNN, B, cnn_args)
                params = trainer.init_model(key, x)
                trainer.benchmark(params, x, y)
                
            except Exception as e:
                # Log the exception to the console
                print(f"Error during benchmark with B={B} and N={N}: {e}")
                # Log the exception to wandb
                #wandb.log({"error": str(e)})
