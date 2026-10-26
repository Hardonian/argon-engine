"""Google DeepMind / JAX Flax Linen Neural Module for Argon Engine.

Implements Forward-Forward SRAM tile plasticity conforming to ADR-0001
for researchers developing with Flax Linen and JAX on Google Cloud TPUs/GPUs.
"""

from typing import Any, Callable, Dict, Optional, Tuple, Union

try:
    import jax
    import jax.numpy as jnp
    from flax import linen as nn
    HAS_FLAX = True
except ImportError:
    import numpy as jnp
    HAS_FLAX = False
    nn = None


def default_diagonal_init(key, shape, dtype=jnp.float32):
    """Canonical diagonal initialization for persistent SRAM tiles."""
    dim = shape[0]
    diag = jnp.eye(dim, dtype=dtype) * 0.5
    return diag


if HAS_FLAX:
    class ArgonFlaxLayer(nn.Module):
        """Flax Linen module implementing an Argon Forward-Forward neural tile.

        Attributes:
            features: Dimensionality of the SRAM tile (default 8).
            learning_rate: Base plasticity rate (default 0.02).
            goodness_threshold: Target homeostatic goodness threshold (default 1.0).
            weight_decay: L2 synaptic weight decay factor (default 0.0005).
            kernel_init: Weight initializer callable.
        """
        features: int = 8
        learning_rate: float = 0.02
        goodness_threshold: float = 1.0
        weight_decay: float = 0.0005
        kernel_init: Callable = default_diagonal_init

        @nn.compact
        def __call__(
            self,
            x: jnp.ndarray,
            polarity: Optional[float] = None,
            training: bool = False,
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """Execute a Forward-Forward inference or plasticity step.

            Args:
                x: Input activation vector of shape `(..., features)`.
                polarity: Contrastive phase scalar (+1.0 positive, -1.0 negative).
                training: Whether to compute and record plasticity delta in variables.

            Returns:
                Tuple `(normalized_output, goodness)` where:
                - `normalized_output` has shape `(..., features)`
                - `goodness` is the mean squared activation scalar per sample.
            """
            weights = self.param(
                "kernel",
                self.kernel_init,
                (self.features, self.features),
                x.dtype,
            )

            # 1. Forward linear drive + ReLU activation
            accum = jnp.dot(x, weights.T)
            y = jnp.maximum(accum, 0.0)

            # 2. Mean squared goodness per sample
            sum_squared = jnp.sum(y * y, axis=-1, keepdims=True)
            goodness = sum_squared / float(self.features)

            # 3. RMS normalization
            inv_rms = 1.0 / (jnp.sqrt(goodness) + 1e-6)
            normalized_output = y * inv_rms

            # 4. Optional plasticity computation during training phase
            if training and polarity is not None:
                margin = polarity * (self.goodness_threshold - goodness)
                modulation = 1.0 / (1.0 + jnp.exp(-margin))
                effective_lr = self.learning_rate * polarity * modulation
                post_drive = effective_lr * y
                
                # Compute batch-averaged delta_w: outer product of post_drive and input
                if x.ndim == 1:
                    delta_w = jnp.outer(post_drive, x) - self.weight_decay * weights
                else:
                    # Batched outer product: (D, B) @ (B, D) -> (D, D)
                    delta_w = jnp.matmul(post_drive.T, x) / float(x.shape[0]) - self.weight_decay * weights
                
                # Record plasticity delta in 'plasticity' variable collection
                delta_var = self.variable("plasticity", "delta_w", lambda: jnp.zeros_like(weights))
                delta_var.value = delta_w

            return normalized_output, jnp.squeeze(goodness, axis=-1)

else:
    class ArgonFlaxLayer:
        """Standalone fallback implementation when Flax is not installed.

        Implements the exact same forward-forward computation and signature
        using pure NumPy for testing, evaluation, and CPU execution.
        """
        def __init__(
            self,
            features: int = 8,
            learning_rate: float = 0.02,
            goodness_threshold: float = 1.0,
            weight_decay: float = 0.0005,
            kernel_init: Optional[Callable] = None,
        ):
            self.features = features
            self.learning_rate = learning_rate
            self.goodness_threshold = goodness_threshold
            self.weight_decay = weight_decay
            self.weights = jnp.eye(features, dtype=jnp.float32) * 0.5
            self.last_delta_w = jnp.zeros_like(self.weights)

        def __call__(
            self,
            x: jnp.ndarray,
            polarity: Optional[float] = None,
            training: bool = False,
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            accum = jnp.dot(x, self.weights.T)
            y = jnp.maximum(accum, 0.0)

            sum_squared = jnp.sum(y * y, axis=-1, keepdims=True)
            goodness = sum_squared / float(self.features)

            inv_rms = 1.0 / (jnp.sqrt(goodness) + 1e-6)
            normalized_output = y * inv_rms

            if training and polarity is not None:
                margin = polarity * (self.goodness_threshold - goodness)
                modulation = 1.0 / (1.0 + jnp.exp(-margin))
                effective_lr = self.learning_rate * polarity * modulation
                post_drive = effective_lr * y

                if x.ndim == 1:
                    delta_w = jnp.outer(post_drive, x) - self.weight_decay * self.weights
                else:
                    # Batched outer product: (D, B) @ (B, D) -> (D, D)
                    delta_w = (post_drive.T @ x) / float(x.shape[0]) - self.weight_decay * self.weights

                self.last_delta_w = delta_w
                self.weights = self.weights + delta_w

            return normalized_output, jnp.squeeze(goodness, axis=-1)
