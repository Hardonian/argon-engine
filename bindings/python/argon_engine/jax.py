"""Pure functional JAX / OpenXLA / TPU Forward-Forward module for Argon Engine.

Conforms to ADR-0001 semantics and numerical tolerances in docs/numerics.md.
Supports `jax.jit`, `jax.vmap`, and `jax.grad` for TPU/GPU acceleration.
Falls back seamlessly to NumPy when JAX is not installed in the active environment.
"""

from dataclasses import dataclass
from typing import Tuple, Union

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    import numpy as jnp  # Fallback to standard NumPy
    HAS_JAX = False


@dataclass(frozen=True)
class ArgonJaxConfig:
    """Hyperparameters for persistent SRAM Forward-Forward plasticity."""
    tile_dim: int = 8
    learning_rate: float = 0.02
    goodness_threshold: float = 1.0
    weight_decay: float = 0.0005


def forward_forward_step(
    weights: jnp.ndarray,
    sensory_input: jnp.ndarray,
    polarity: float,
    learning_rate: float,
    goodness_threshold: float,
    weight_decay: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
    """Pure, JIT-traceable Forward-Forward execution step.

    Parameters:
        weights: 2D array of shape (tile_dim, tile_dim)
        sensory_input: 1D array of shape (tile_dim,)
        polarity: Phase polarity scalar (1.0 for positive, -1.0 for negative)
        learning_rate: Base plasticity rate
        goodness_threshold: Target homeostatic goodness threshold
        weight_decay: L2 synaptic weight decay factor

    Returns:
        (new_weights, normalized_output, goodness)
    """
    dim = weights.shape[0]

    # 1. Forward linear drive + ReLU activation
    accum = jnp.dot(weights, sensory_input)
    y = jnp.maximum(accum, 0.0)

    # 2. Mean squared goodness across SRAM tile
    sum_squared = jnp.sum(y * y)
    goodness = sum_squared / float(dim)

    # 3. Logistic modulation margin
    margin = polarity * (goodness_threshold - goodness)
    modulation = 1.0 / (1.0 + jnp.exp(-margin))
    effective_lr = learning_rate * polarity * modulation

    # 4. Normalized output activations (RMS normalization)
    inv_rms = 1.0 / (jnp.sqrt(goodness) + 1e-6)
    normalized_output = y * inv_rms

    # 5. Local Hebbian synaptic update + weight decay
    post_drive = effective_lr * y
    delta_w = jnp.outer(post_drive, sensory_input) - weight_decay * weights
    new_weights = weights + delta_w

    return new_weights, normalized_output, goodness


# Compile with jax.jit if JAX runtime is present
if HAS_JAX:
    forward_forward_step_jit = jax.jit(forward_forward_step, static_argnames=())
else:
    forward_forward_step_jit = forward_forward_step


class ArgonJaxTile:
    """Stateful wrapper managing a single Forward-Forward tile with JAX/NumPy."""

    def __init__(
        self,
        config: ArgonJaxConfig = ArgonJaxConfig(),
        initial_weights: Union[jnp.ndarray, None] = None,
        use_jit: bool = True,
    ):
        self.config = config
        self.dim = config.tile_dim
        self.use_jit = use_jit and HAS_JAX

        if initial_weights is not None:
            self.weights = jnp.array(initial_weights, dtype=jnp.float32).reshape((self.dim, self.dim))
        else:
            # Default canonical diagonal initialization
            diag = jnp.eye(self.dim, dtype=jnp.float32) * 0.45 + 0.02
            self.weights = diag

        self.cumulative_steps = 0

    def step(self, sensory_input: jnp.ndarray, polarity: float = 1.0) -> Tuple[jnp.ndarray, float]:
        """Executes one Forward-Forward step, updating internal resident weights."""
        step_fn = forward_forward_step_jit if self.use_jit else forward_forward_step
        new_w, out, goodness = step_fn(
            self.weights,
            sensory_input,
            polarity,
            self.config.learning_rate,
            self.config.goodness_threshold,
            self.config.weight_decay,
        )
        self.weights = new_w
        self.cumulative_steps += 1
        return out, float(goodness)

    def get_weights_flat(self) -> jnp.ndarray:
        """Returns flattened 1D array of weights."""
        return self.weights.flatten()
