import os
import sys
from types import SimpleNamespace
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from argon_engine import ArgonEngine
from argon_engine.grpc_service import ArgonGrpcServicer


def test_grpc_servicer_step_and_checksum():
    engine = ArgonEngine(num_layers=2, tile_dim=8)
    servicer = ArgonGrpcServicer(engine)

    # Test Step RPC
    req = SimpleNamespace(
        features=[0.5, 0.4, 0.3, 0.2, 0.1, 0.2, 0.3, 0.4],
        phase_polarity=1.0,
    )
    res = servicer.Step(req)
    assert res["sequence_id"] == 1
    assert len(res["activations"]) == 8
    assert res["energy_l2"] > 0.0
    assert res["telemetry"]["mean_goodness"] > 0.0

    # Test GetWeightChecksum RPC
    ck_res = servicer.GetWeightChecksum()
    assert ck_res["checksum"] != 0

    # Test ScaleWeights RPC
    scale_req = SimpleNamespace(target_norm=2.0)
    scale_res = servicer.ScaleWeights(scale_req)
    assert scale_res["success"] is True
    assert scale_res["new_checksum"] != 0

    # Test ExportWeights RPC
    exp_res = servicer.ExportWeights()
    assert len(exp_res["weights"]) == 2 * 8 * 8
    assert exp_res["tile_dim"] == 8
    assert exp_res["num_layers"] == 2

    engine.close()


def test_grpc_servicer_bi_directional_stream():
    engine = ArgonEngine(num_layers=2, tile_dim=8)
    servicer = ArgonGrpcServicer(engine)

    def request_generator():
        for i in range(3):
            yield SimpleNamespace(
                frame=SimpleNamespace(
                    features=[0.1 * (i + 1)] * 8,
                    phase_polarity=1.0,
                ),
                scale=None,
            )
        yield SimpleNamespace(
            frame=None,
            scale=SimpleNamespace(target_norm=1.5),
        )

    responses = list(servicer.StreamClosedLoop(request_generator()))
    assert len(responses) == 4
    assert "action" in responses[0]
    assert "action" in responses[1]
    assert "action" in responses[2]
    assert "scale_result" in responses[3]
    assert responses[3]["scale_result"]["success"] is True

    engine.close()
