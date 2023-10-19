from typing import Sequence

import jax.numpy as jnp
from flax import linen as nn

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

