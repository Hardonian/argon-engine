import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from argon_engine.jax import ArgonJaxConfig, ArgonJaxTile, forward_forward_step


def test_jax_tile_golden_vector_parity():
    # Canonical ADR-0001 golden-vector parameters:
    # 4x4 tile with diagonal 0.5, input [0.1, 0.2, 0.3, 0.4]
    # learning_rate=0.01, goodness_threshold=1.0, weight_decay=0.001, polarity=1.0
    config = ArgonJaxConfig(
        tile_dim=4,
        learning_rate=0.01,
        goodness_threshold=1.0,
        weight_decay=0.001,
    )
    initial_weights = np.zeros((4, 4), dtype=np.float32)
    for i in range(4):
        initial_weights[i, i] = 0.5

    tile = ArgonJaxTile(config=config, initial_weights=initial_weights, use_jit=False)
    sensory = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    output, goodness = tile.step(sensory, polarity=1.0)

    # 1. Verify goodness
    assert abs(goodness - 0.01875) < 1.0e-6

    # 2. Verify golden vector output activations: [0.3651457, 0.7302914, 1.0954372, 1.4605829]
    golden = np.array([0.3651457, 0.7302914, 1.0954372, 1.4605829], dtype=np.float32)
    np.testing.assert_allclose(output, golden, atol=2.0e-6)

    # 3. Verify weight update
    weights = tile.weights
    assert abs(weights[0, 0] - 0.49953637) < 2.0e-7
    assert abs(weights[0, 1] - 0.00007273562) < 2.0e-9


def test_jax_tile_plasticity_monotonic_increase():
    # Positive phase should increase goodness over repeated ticks
    config = ArgonJaxConfig(tile_dim=8, learning_rate=0.05, goodness_threshold=3.0)
    tile = ArgonJaxTile(config=config)

    sensory = np.array([0.9, 0.4, 0.7, 0.3, 0.8, 0.5, 0.6, 0.2], dtype=np.float32)
    _, first_goodness = tile.step(sensory, polarity=1.0)

    last_goodness = first_goodness
    for _ in range(8):
        _, last_goodness = tile.step(sensory, polarity=1.0)

    assert last_goodness > first_goodness
    assert tile.cumulative_steps == 9
