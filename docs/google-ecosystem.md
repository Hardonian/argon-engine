# Google Developer Ecosystem Integration Guide

Argon Engine integrates natively with the Google developer, cloud, research, and edge ecosystem across six core dimensions:

---

## 1. Google DeepMind & Research: JAX & TPU Integration

Argon provides a pure functional JAX and OpenXLA tile implementation in [`bindings/python/argon_engine/jax.py`](file:///c:/Users/scott/argon-engine/bindings/python/argon_engine/jax.py) adhering strictly to ADR-0001 semantics and numerical tolerances in `docs/numerics.md`.

- **JIT Compilation**: Compatible with `jax.jit` for zero-overhead compilation to Google Cloud TPUs (v4/v5e/v5p/v6e Trillium) and GPU accelerators.
- **Batching & Vectorization**: Native support for `jax.vmap` across sensory streams.
- **Gradient Composition**: Differentiable forward path while executing local Forward-Forward / Hebbian synaptic updates.

```python
import jax
import jax.numpy as jnp
from argon_engine.jax import ArgonJaxTile, ArgonJaxConfig

config = ArgonJaxConfig(tile_dim=8, learning_rate=0.02, goodness_threshold=1.0)
tile = ArgonJaxTile(config=config)

sensory = jnp.array([0.5, 0.4, 0.3, 0.2, 0.1, 0.2, 0.3, 0.4], dtype=jnp.float32)
output, goodness = tile.step(sensory, polarity=1.0)
```

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
