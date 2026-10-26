# Argon Engine: Google Developer Ecosystem Integration Guide

This guide details how Google software engineers, research scientists (DeepMind / Google Research), and cloud infrastructure architects can integrate and consume **Argon Engine** natively within the Google developer toolchain.

---

## 1. Hermetic Bazel Build System (Bzlmod)

Argon Engine provides standard Bazel build targets defined in [`MODULE.bazel`](../MODULE.bazel) and [`BUILD.bazel`](../BUILD.bazel).

### Consuming in your `BUILD` file

```python
# Modern C++20 RAII targets
cc_binary(
    name = "neuromorphic_service",
    srcs = ["service.cc"],
    deps = [
        "@argon//:argon_cpp",
    ],
)

# Python zero-copy targets
py_binary(
    name = "train_flax",
    srcs = ["train_flax.py"],
    deps = [
        "@argon//:argon_python",
    ],
)

# Protobuf / gRPC streaming targets
proto_library(
    name = "my_service_proto",
    srcs = ["service.proto"],
    deps = [
        "@argon//:argon_proto",
    ],
)
```

---

## 2. Google-Style C++20 RAII Header (`include/argon.hpp`)

Conforming strictly to the **Google C++ Style Guide**, Argon's C++20 wrapper provides:

- **`std::span` Views**: Zero heap allocation, zero pointer-length arithmetic bugs.
- **Value-or-Status Semantics**: `argon::Result<T>` and `argon::Status` mimicking `absl::StatusOr<T>`.
- **Automatic RAII Destruction**: Deterministic coordinator reclamation when scopes exit.

### C++20 Usage Example

```cpp
#include <span>
#include <vector>
#include "argon.hpp"

void RunInference() {
  // 1. Initialize 2 layers, 8-dimensional SRAM tile
  auto engine_res = argon::Engine::Create(
      /*num_layers=*/2,
      /*tile_dim=*/8,
      /*learning_rate=*/0.02f,
      /*goodness_threshold=*/1.0f,
      /*weight_decay=*/0.0005f,
      /*ring_capacity=*/8);
  if (!engine_res.Ok()) {
    // Handle error via Google-style status message
    return;
  }
  auto engine = std::move(engine_res.Value());

  // 2. Execute zero-copy forward tick with std::span
  const std::vector<float> input = {0.5f, 0.4f, 0.3f, 0.2f, 0.1f, 0.2f, 0.3f, 0.4f};
  std::vector<float> output(8, 0.0f);

  auto step_res = engine.Step(/*sequence_id=*/1, /*polarity=*/1.0f, input, output);
  if (step_res.Ok()) {
    float goodness = step_res->mean_goodness;
    // Output activations are populated directly in `output` with zero copies
  }

  // 3. Bit-exact weight snapshot & checksum
  uint64_t checksum = engine.WeightChecksum().Value();
  auto saved_weights = engine.ExportWeights().Value();
}
```

---

## 3. Google DeepMind / JAX Flax Linen Neural Module (`argon_engine.flax`)

For JAX/XLA researchers on Google Cloud TPUs, `ArgonFlaxLayer` integrates with Flax Linen (`flax.linen.Module`):

```python
import jax
import jax.numpy as jnp
from argon_engine.flax import ArgonFlaxLayer

# Instantiate Flax Linen layer
layer = ArgonFlaxLayer(features=8, learning_rate=0.02, goodness_threshold=1.0)

# Forward pass (JIT-traceable)
x = jnp.ones((4, 8), dtype=jnp.float32)
output, goodness = layer(x, training=False)

# Contrastive forward-forward training step (+1.0 positive phase)
train_out, train_goodness = layer(x, polarity=1.0, training=True)
```

---

## 4. Google Keras 3 Multi-Backend Layer (`argon_engine.keras`)

For production ML engineers using Keras 3 (across JAX, TensorFlow, or PyTorch backends):

```python
import keras
from argon_engine.keras import ArgonKerasLayer

model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    ArgonKerasLayer(units=8, learning_rate=0.02, goodness_threshold=1.0),
    keras.layers.Dense(2, activation="softmax"),
])

# Standard Keras model compilation & serialization
model.compile(optimizer="adam", loss="categorical_crossentropy")
```

---

## 5. Google Cloud Trace & OpenTelemetry (`argon_engine.otel`)

Argon Engine instruments all inference ticks with OpenTelemetry spans compatible with **Google Cloud Trace**:

```python
import numpy as np
from argon_engine import ArgonEngine
from argon_engine.otel import ArgonTelemetryTracer, setup_google_cloud_trace_exporter

# Automatically route spans to Google Cloud Trace if credentials are present
setup_google_cloud_trace_exporter(project_id="my-gcp-project")

tracer = ArgonTelemetryTracer(service_name="argon-inference-service")

with ArgonEngine(num_layers=2, tile_dim=8) as engine:
    sensory = np.array([0.1] * 8, dtype=np.float32)
    # Automatically emits latency_ns, mean_goodness, and sequence_id spans
    out, telemetry = tracer.instrument_engine_step(
        engine=engine,
        sensory_input=sensory,
        polarity=1.0,
    )
```

---

## 6. Vertex AI & Cloud Run Distroless Deployment

Argon includes production-grade container manifests for Google Cloud:

- [`deploy/Dockerfile.vertex`](../deploy/Dockerfile.vertex): Minimal, non-root, distroless image containing pre-compiled zero-copy shared libraries and health check probes.
- [`deploy/vertex_handler.py`](../deploy/vertex_handler.py): HTTP prediction handler conforming to Vertex AI Custom Container specifications (`/predict` and `/health` endpoints).

### Building and deploying to Google Artifact Registry

```bash
docker build -t gcr.io/${PROJECT_ID}/argon-vertex:latest -f deploy/Dockerfile.vertex .
docker push gcr.io/${PROJECT_ID}/argon-vertex:latest
```

---

## 7. Performance & Latency Envelope

| Component | Target Metric | Measured (Local) |
| :--- | :--- | :--- |
| SIMD Closed-Loop Step | `< 1,000,000 ns` (1 ms) | **`1,200 ns`** (~0.0012 ms) |
| Warmed Allocations | `0 bytes` | **`0 bytes`** (verified via `zero_allocation` test) |
| C++20 Call Overhead | `< 50 ns` | Zero-copy via `std::span` |
| Serialization | Bit-exact | `xxHash64` bit-exact verification |
