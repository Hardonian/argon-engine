"""Google Cloud Vertex AI & Cloud Run custom prediction HTTP handler.

Conforms strictly to the Google Cloud Vertex AI custom prediction container contract:
- Health check: GET /healthz or GET /readyz (returns 200 OK)
- Prediction route: POST /predict (takes {"instances": [...]}, returns {"predictions": [...]})
- Port configuration: respects AIP_HTTP_PORT and PORT environment variables (default: 8080)
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np

# Ensure bindings/python is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bindings", "python")))

from argon_engine import ArgonEngine

ENGINE: ArgonEngine = None


def get_engine() -> ArgonEngine:
    global ENGINE
    if ENGINE is None:
        ENGINE = ArgonEngine(num_layers=2, tile_dim=8)
    return ENGINE


class VertexHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Vertex AI / Cloud Run liveness and readiness probe."""
        if self.path in ("/healthz", "/readyz", "/health", "/"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            engine = get_engine()
            payload = {
                "status": "healthy",
                "service": "argon-engine",
                "tile_dim": engine.tile_dim,
                "num_layers": engine.num_layers,
                "weight_checksum": hex(engine.weight_checksum()),
            }
            self.wfile.write(json.dumps(payload).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        """Vertex AI custom prediction endpoint."""
        if self.path in ("/predict", "/"):
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode("utf-8"))
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Invalid JSON payload: {str(e)}"}).encode("utf-8"))
                return

            instances = data.get("instances", [])
            engine = get_engine()
            predictions = []

            for instance in instances:
                if isinstance(instance, dict):
                    features = instance.get("features", [])
                    polarity = float(instance.get("polarity", 1.0))
                elif isinstance(instance, list):
                    features = instance
                    polarity = 1.0
                else:
                    continue

                sensory = np.array(features, dtype=np.float32)
                if sensory.shape != (engine.tile_dim,):
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(
                        json.dumps({
                            "error": f"Input dimension mismatch: expected {engine.tile_dim}, got {sensory.shape[0]}"
                        }).encode("utf-8")
                    )
                    return

                output, telemetry = engine.step(sensory, polarity=polarity)
                predictions.append({
                    "sequence_id": telemetry.sequence_id,
                    "activations": output.tolist(),
                    "mean_goodness": telemetry.mean_goodness,
                    "output_l2_norm": telemetry.output_l2_norm,
                    "total_ticks": telemetry.total_ticks,
                })

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"predictions": predictions}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Override to suppress default stdout log spam
        pass


def run_server():
    port = int(os.environ.get("AIP_HTTP_PORT", os.environ.get("PORT", "8080")))
    server_address = ("0.0.0.0", port)
    httpd = HTTPServer(server_address, VertexHandler)
    print(f"Argon Vertex AI prediction server listening on port {port}...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    run_server()
