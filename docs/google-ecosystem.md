# Google Developer Ecosystem Integration Guide

Argon Engine integrates natively with the Google developer, cloud, research, and edge ecosystem across six core dimensions:

---

## 1. Google DeepMind & Research: JAX, Flax Linen & TPU Integration

Argon provides pure functional JAX and Flax Linen implementations in [`bindings/python/argon_engine/jax.py`](file:///c:/Users/scott/argon-engine/bindings/python/argon_engine/jax.py) and [`bindings/python/argon_engine/flax.py`](file:///c:/Users/scott/argon-engine/bindings/python/argon_engine/flax.py) adhering strictly to ADR-0001 semantics and numerical tolerances in `docs/numerics.md`.

- **Flax Linen Neural Module**: `ArgonFlaxLayer` integrates into Flax `nn.Module` networks with `@nn.compact` and parameter dictionaries.
- **JIT Compilation**: Compatible with `jax.jit` for zero-overhead compilation to Google Cloud TPUs (v4/v5e/v5p/v6e Trillium) and GPU accelerators.
- **Batching & Vectorization**: Native support for `jax.vmap` across sensory streams.
- **Gradient Composition**: Differentiable forward path while executing local Forward-Forward / Hebbian synaptic updates.

```python
import jax.numpy as jnp
from argon_engine.flax import ArgonFlaxLayer

layer = ArgonFlaxLayer(features=8, learning_rate=0.02, goodness_threshold=1.0)
x = jnp.ones((4, 8), dtype=jnp.float32)
output, goodness = layer(x, training=False)
```

---

## 2. Google Core Systems: Modern C++20 RAII Library (`include/argon.hpp`)

Conforming strictly to the **Google C++ Style Guide**, Argon's C++20 library provides:

- **`std::span` Zero-Copy Views**: Memory safety without raw pointer arithmetic.
- **Value-or-Status Outcome Semantics**: `argon::Result<T>` and `argon::Status` matching `absl::StatusOr<T>`.
- **RAII Deterministic Reclamation**: Lifetime managed by `argon::Engine`.

```cpp
#include <span>
#include <vector>
#include "argon.hpp"

auto engine = argon::Engine::Create(2, 8, 0.02f, 1.0f, 0.0005f, 8).Value();
const std::vector<float> input = {0.5f, 0.4f, 0.3f, 0.2f, 0.1f, 0.2f, 0.3f, 0.4f};
std::vector<float> output(8, 0.0f);
auto step_res = engine.Step(1, 1.0f, input, output);
```

---

## 3. Google Keras 3 Multi-Backend & OpenTelemetry Observability

- **Keras 3 Multi-Backend**: [`bindings/python/argon_engine/keras.py`](file:///c:/Users/scott/argon-engine/bindings/python/argon_engine/keras.py) provides `ArgonKerasLayer`, serializable and runnable on JAX, TensorFlow, or PyTorch backends.
- **Google Cloud Trace & OpenTelemetry**: [`bindings/python/argon_engine/otel.py`](file:///c:/Users/scott/argon-engine/bindings/python/argon_engine/otel.py) provides `ArgonTelemetryTracer` capturing per-tick nanosecond latency and goodness metrics directly for Cloud Trace.

---

## 2. Google Cloud Platform (GCP): Vertex AI & Cloud Run

Argon provides a production container definition in [`deploy/Dockerfile.vertex`](file:///c:/Users/scott/argon-engine/deploy/Dockerfile.vertex) and custom prediction server in [`deploy/vertex_handler.py`](file:///c:/Users/scott/argon-engine/deploy/vertex_handler.py).

- **Vertex AI Contract**:
  - `GET /healthz`: Health and readiness probe returning engine topology and weight checksum.
  - `POST /predict`: Custom prediction endpoint accepting standard `{"instances": [...]}` payloads.
- **Cloud Run Port Binding**: Automatically detects `$AIP_HTTP_PORT` and `$PORT` (default 8080).
- **Distroless & Non-Root**: Built as a minimal multi-stage image running as non-root user `65534` for GCP security compliance.

---

## 3. Google Standard IPC: Protocol Buffers & gRPC

Argon defines a canonical Protobuf v3 schema in [`proto/argon/v1/engine.proto`](file:///c:/Users/scott/argon-engine/proto/argon/v1/engine.proto) and Python servicer in [`bindings/python/argon_engine/grpc_service.py`](file:///c:/Users/scott/argon-engine/bindings/python/argon_engine/grpc_service.py).

- **Bi-directional Streaming**: `rpc StreamClosedLoop(stream StreamClosedLoopRequest) returns (stream StreamClosedLoopResponse)` allows real-time hardware-in-the-loop streaming with zero per-tick allocations.
- **Cross-Language Client SDKs**: Compile the `.proto` file to Go, Java/Kotlin, C++, Python, or Dart/Flutter using `protoc` or Bazel.

---

## 4. Google Colab: One-Click Interactive Quickstart

Argon features an interactive quickstart notebook in [`notebooks/argon_quickstart.ipynb`](file:///c:/Users/scott/argon-engine/notebooks/argon_quickstart.ipynb):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hardonian/argon-engine/blob/main/notebooks/argon_quickstart.ipynb)

Includes interactive demonstrations of:

- ADR-0001 golden-vector verification.
- Contrastive Forward-Forward goodness separation plots.
- Closed-loop sensory-actuator oscillation damping.

---

## 5. Google Monorepos: Hermetic Bazel Builds

Argon supports modern Bazel 7+ Bzlmod via [`MODULE.bazel`](file:///c:/Users/scott/argon-engine/MODULE.bazel) and [`BUILD.bazel`](file:///c:/Users/scott/argon-engine/BUILD.bazel):

- `//:argon_c_header`: C-ABI headers for C/C++ Google monorepo targets.
- `//:argon_proto`: Canonical `proto_library` target.
- `//:argon_python`: Hermetic Python library target.

---

## 6. Edge & Android: Android NDK Integration

Argon compiles directly to `libargon_ffi.so` for Android NDK architectures (`arm64-v8a`, `armeabi-v7a`, `x86_64`):

- CMake configuration: [`bindings/android/CMakeLists.txt`](file:///c:/Users/scott/argon-engine/bindings/android/CMakeLists.txt)
- JNI C bridge: [`bindings/android/src/argon_jni.c`](file:///c:/Users/scott/argon-engine/bindings/android/src/argon_jni.c)
- Pixel (Tensor G3/G4) integration guide: [`bindings/android/README.md`](file:///c:/Users/scott/argon-engine/bindings/android/README.md)

---

## 7. Agentic AI: Model Context Protocol (MCP) Server for Gemini & Antigravity

Argon includes a native JSON-RPC 2.0 stdio MCP server in [`scripts/argon_mcp_server.py`](file:///c:/Users/scott/argon-engine/scripts/argon_mcp_server.py).

Enables Gemini and Google Antigravity agents to autonomously:

- Probe local hardware and GPU acceleration backends (`argon_inspect`).
- Verify cross-backend bit-exact golden vectors (`argon_verify_parity`).
- Execute inference and plasticity steps (`argon_step`).
- Query weight checksums and rescale homeostatic norms (`argon_weight_checksum`, `argon_scale_weights`).
