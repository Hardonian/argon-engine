# Backend qualification

| Backend | Availability | Qualification |
| --- | --- | --- |
| Rust scalar CPU | Production reference | Workspace tests and deterministic replay |
| WASM scalar/SIMD layout | Supported | WASM build, native parity tests, TypeScript harness |
| NumPy dense reference | Supported | Deterministic Python execution |
| Vulkan compute | Supported (P2) | GLSL 450 shader, numerical contract, golden-vector parity |
| Apple Metal compute | Supported (P2) | MSL compute kernel, numerical contract, golden-vector parity |
| AMD ROCm/HIP | Supported (P2) | HIP C++ compute kernel, numerical contract, golden-vector parity |
| Triton GPU | Experimental | Requires CUDA runner and parity workflow |
| Persistent PTX | Prototype | Not selected by runtime until driver integration and parity pass |
| Local model servers | Supported integration | Loopback-only health check, streaming, and native API contract |

An experimental backend is never advertised as active merely because source is
present. The GPU workflow records device information and fails if CUDA or Triton
is absent on its explicitly labelled runner.

## Consumer GPU, non-CUDA accelerators, and sovereign-local policy

Argon always keeps the scalar Rust runtime available. Non-CUDA hardware acceleration
(Vulkan, Apple Metal, AMD ROCm) is implemented with verified resident compute execution engines
complying with the numerical contract (`< 2e-5` single-tick, `< 2e-4` over 1,000 ticks) and
bit-exact golden vectors.

NVIDIA GPUs are detected through `nvidia-smi` for inventory and sizing only; discovery
never claims that the experimental PTX backend has been selected. The CLI maps reported VRAM to
conservative tiers: under 6 GiB (small quantized models), 6–11 GiB (7B/8B
quantized), 12–23 GiB (8B–14B quantized), and 24 GiB or more (larger quantized
models and longer contexts).

`argon-backend::LocalAiStack` connects to Ollama, llama.cpp, LM Studio, and
self-hosted OpenAI-compatible APIs. It accepts only `http://` loopback URLs by default,
uses no cloud API key by default, and maintains credential-free local execution.
Streaming token emission (`complete_stream`), authenticated endpoints with Bearer tokens,
and model lifecycle operations (`pull_model`, `unload_model`, `model_info`) are available
as explicit opt-in features.
