import os
import sys
import numpy as np
import pytest

# Ensure bindings/python is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from argon_engine import ArgonEngine, TickTelemetry


def test_engine_lifecycle_and_zero_copy():
    with ArgonEngine(num_layers=2, tile_dim=8, learning_rate=0.02) as engine:
        sensory = np.array([0.5, 0.4, 0.3, 0.2, 0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        out_buf = np.zeros(8, dtype=np.float32)

        res, telemetry = engine.step(sensory, polarity=1.0, output=out_buf)

        assert res is out_buf
        assert telemetry.sequence_id == 1
        assert telemetry.total_ticks == 1
        assert telemetry.mean_goodness > 0.0
        assert telemetry.output_l2_norm > 0.0
        assert np.any(out_buf != 0.0)


def test_engine_golden_vector_parity():
    # Matches the exact canonical dense forward golden vector test from argon-kernel:
    # 4x4 tile with diagonal 0.5, learning_rate=0.01, goodness_threshold=1.0, weight_decay=0.001
    with ArgonEngine(
        num_layers=1,
        tile_dim=4,
        learning_rate=0.01,
        goodness_threshold=1.0,
        weight_decay=0.001,
        ring_capacity=4,
    ) as engine:
        weights = np.zeros(16, dtype=np.float32)
        for i in range(4):
            weights[i * 4 + i] = 0.5
        engine.restore_weights(weights)

        sensory = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        out, telemetry = engine.step(sensory, polarity=1.0)

        golden = np.array([0.3651457, 0.7302914, 1.0954372, 1.4605829], dtype=np.float32)
        np.testing.assert_allclose(out, golden, atol=2e-6)
        assert abs(telemetry.mean_goodness - 0.01875) < 1e-6


def test_weight_checksum_export_scale_restore():
    with ArgonEngine(num_layers=2, tile_dim=8) as engine:
        initial_checksum = engine.weight_checksum()
        assert initial_checksum != 0

        exported = engine.export_weights()
        assert exported.shape == (2 * 8 * 8,)

        engine.scale_weights(2.0)
        scaled_checksum = engine.weight_checksum()
        assert scaled_checksum != initial_checksum

        engine.restore_weights(exported)
        restored_checksum = engine.weight_checksum()
        assert restored_checksum == initial_checksum


def test_dimension_mismatch_raises():
    with ArgonEngine(num_layers=1, tile_dim=8) as engine:
        bad_input = np.ones(4, dtype=np.float32)
        with pytest.raises(ValueError, match="shape mismatch"):
            engine.step(bad_input)


def test_attestation_hash_and_integrity_verification():
    with ArgonEngine(num_layers=2, tile_dim=8) as engine:
        att_hash = engine.attestation_hash()
        assert att_hash != 0
        assert isinstance(att_hash, int)

        assert engine.verify_integrity() is True

        # After step, attestation updates deterministically
        sensory = np.ones(8, dtype=np.float32) * 0.5
        engine.step(sensory)
        stepped_att_hash = engine.attestation_hash()
        assert stepped_att_hash != att_hash
        assert engine.verify_integrity() is True
