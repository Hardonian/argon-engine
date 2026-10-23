# Backend qualification

| Backend | Availability | Qualification |
| --- | --- | --- |
| Rust scalar CPU | Production reference | Workspace tests and deterministic replay |
| WASM scalar/SIMD layout | Supported | WASM build, native parity tests, TypeScript harness |
| NumPy dense reference | Supported | Deterministic Python execution |
| Triton GPU | Experimental | Requires CUDA runner and parity workflow |
| Persistent PTX | Prototype | Not selected by runtime until driver integration and parity pass |
| Local model servers | Supported integration | Loopback-only health check and native completion API contract |

An experimental backend is never advertised as active merely because source is
present. The GPU workflow records device information and fails if CUDA or Triton
is absent on its explicitly labelled runner.

## Consumer GPU and sovereign-local policy

Argon always keeps the scalar Rust runtime available. NVIDIA GPUs are detected
through `nvidia-smi` for inventory and sizing only; discovery never claims that
the experimental PTX backend has been selected. The CLI maps reported VRAM to
conservative tiers: under 6 GiB (small quantized models), 6–11 GiB (7B/8B
quantized), 12–23 GiB (8B–14B quantized), and 24 GiB or more (larger quantized
models and longer contexts).

`argon-backend::LocalAiStack` connects to Ollama, llama.cpp, and self-hosted
OpenAI-compatible APIs. It accepts only `http://` loopback URLs by default,
uses no cloud API key, and does not start or download models. A caller must
explicitly opt in before using a LAN or remote endpoint. This keeps model data,
prompts, and inference traffic local unless the embedding application makes a
different deliberate policy choice.
