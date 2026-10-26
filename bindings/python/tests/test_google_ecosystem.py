"""Comprehensive tests for Google Developer Ecosystem modules:

1. Flax Linen neural module (Google DeepMind / JAX)
2. Keras 3 multi-backend layer (Google Keras / TensorFlow / JAX)
3. OpenTelemetry distributed tracing (Google Cloud Trace / Cloud Monitoring)
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from argon_engine import (
    ArgonEngine,
    ArgonFlaxLayer,
    ArgonKerasLayer,
    ArgonTelemetryTracer,
)


def test_flax_layer_inference_and_plasticity():
    layer = ArgonFlaxLayer(features=4, learning_rate=0.05, goodness_threshold=2.0)
    x = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32)

    # 1. Forward inference pass (training=False)
    out, goodness = layer(x, training=False)
    assert out.shape == (4,)
    assert goodness > 0.0
    initial_weights = layer.weights.copy()

    # 2. Forward-Forward contrastive plasticity step (positive phase)
    out_train, goodness_train = layer(x, polarity=1.0, training=True)
    assert not np.array_equal(layer.weights, initial_weights)
    assert layer.last_delta_w.shape == (4, 4)

    # 3. Batched execution (batch_size=2)
    batch_x = np.array([
        [0.2, 0.4, 0.6, 0.8],
        [0.1, 0.3, 0.5, 0.7],
    ], dtype=np.float32)
    batch_out, batch_goodness = layer(batch_x, polarity=1.0, training=True)
    assert batch_out.shape == (2, 4)
    assert batch_goodness.shape == (2,)


def test_keras_layer_inference_and_serialization():
    layer = ArgonKerasLayer(units=4, learning_rate=0.03, goodness_threshold=1.5)
    x = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)

    # 1. Inference call
    out = layer(x, training=False)
    assert out.shape == (1, 4)
    assert layer.last_goodness is not None

    # 2. Plasticity step (positive phase)
    initial_kernel = layer.kernel.copy()
    out_pos = layer(x, polarity=1.0, training=True)
    assert not np.array_equal(layer.kernel, initial_kernel)

    # 3. Serialization config check
    config = layer.get_config()
    assert config["units"] == 4
    assert config["learning_rate"] == 0.03
    assert config["goodness_threshold"] == 1.5
    assert config["weight_decay"] == 0.0005


def test_opentelemetry_tracer_context_and_engine_step():
    tracer = ArgonTelemetryTracer(service_name="argon-test-service")

    # 1. Context manager trace span
    with tracer.trace_tick(sequence_id=101, polarity=1.0) as metrics:
        metrics["mean_goodness"] = 1.25
        metrics["output_l2_norm"] = 0.98
        metrics["total_ticks"] = 101

    assert metrics["sequence_id"] == 101
    assert metrics["mean_goodness"] == 1.25

    # 2. Engine step instrumentation
    with ArgonEngine(num_layers=2, tile_dim=8) as engine:
        sensory = np.array([0.1] * 8, dtype=np.float32)
        out, telem = tracer.instrument_engine_step(
            engine=engine,
            sensory_input=sensory,
            polarity=1.0,
            sequence_id=202,
        )
        assert out.shape == (8,)
        assert telem.total_ticks == 1
        assert telem.mean_goodness > 0.0
