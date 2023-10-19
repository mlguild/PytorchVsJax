from typing import Sequence, Any

import jax.numpy as jnp
from flax import linen as nn

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
