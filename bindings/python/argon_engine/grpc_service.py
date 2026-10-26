"""Argon Engine Google gRPC streaming adapter and microservice handler."""

import time
from typing import Generator, Iterable, Optional

import numpy as np

from argon_engine import ArgonEngine, TickTelemetry


class ArgonGrpcServicer:
    """gRPC Servicer bridging ArgonEngine zero-copy native loops with Google gRPC."""

    def __init__(self, engine: Optional[ArgonEngine] = None):
        self.engine = engine or ArgonEngine()

    def Step(self, request, context=None):
        """Synchronous unary RPC executing one Forward-Forward step."""
        sensory = np.array(request.features, dtype=np.float32)
        polarity = request.phase_polarity if request.phase_polarity != 0.0 else 1.0

        output, telemetry = self.engine.step(sensory, polarity=polarity)
        return {
            "sequence_id": telemetry.sequence_id,
            "activations": output.tolist(),
            "energy_l2": telemetry.output_l2_norm,
            "telemetry": {
                "sequence_id": telemetry.sequence_id,
                "mean_goodness": telemetry.mean_goodness,
                "output_l2_norm": telemetry.output_l2_norm,
                "total_ticks": telemetry.total_ticks,
                "queue_depth": 0,
            },
        }

    def GetWeightChecksum(self, request=None, context=None):
        """Unary RPC returning the resident synaptic weight checksum."""
        return {"checksum": self.engine.weight_checksum()}

    def ScaleWeights(self, request, context=None):
        """Unary RPC scaling synaptic weights according to target homeostatic norm."""
        target_norm = float(request.target_norm)
        self.engine.scale_weights(target_norm)
        new_checksum = self.engine.weight_checksum()
        return {"success": True, "new_checksum": new_checksum}

    def ExportWeights(self, request=None, context=None):
        """Unary RPC exporting all layer synaptic weights."""
        weights = self.engine.export_weights()
        checksum = self.engine.weight_checksum()
        return {
            "weights": weights.tolist(),
            "checksum": checksum,
            "tile_dim": self.engine.tile_dim,
            "num_layers": self.engine.num_layers,
        }

    def StreamClosedLoop(
        self, request_iterator: Iterable, context=None
    ) -> Generator[dict, None, None]:
        """Bi-directional streaming RPC for closed-loop sensory-actuator hardware."""
        for req in request_iterator:
            if hasattr(req, "frame") and req.frame:
                frame = req.frame
                sensory = np.array(frame.features, dtype=np.float32)
                polarity = frame.phase_polarity if frame.phase_polarity != 0.0 else 1.0
                out, telemetry = self.engine.step(sensory, polarity=polarity)
                yield {
                    "action": {
                        "sequence_id": telemetry.sequence_id,
                        "activations": out.tolist(),
                        "energy_l2": telemetry.output_l2_norm,
                        "telemetry": {
                            "sequence_id": telemetry.sequence_id,
                            "mean_goodness": telemetry.mean_goodness,
                            "output_l2_norm": telemetry.output_l2_norm,
                            "total_ticks": telemetry.total_ticks,
                            "queue_depth": 0,
                        },
                    }
                }
            elif hasattr(req, "scale") and req.scale:
                target_norm = float(req.scale.target_norm)
                self.engine.scale_weights(target_norm)
                new_ck = self.engine.weight_checksum()
                yield {"scale_result": {"success": True, "new_checksum": new_ck}}
