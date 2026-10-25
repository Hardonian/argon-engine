# Delivery roadmap

## P0 — local-first runtime (implemented)

- Deterministic CPU and WASM execution remain the normative runtime.
- Consumer NVIDIA inventory reports VRAM and conservative local-model guidance
  without falsely selecting an experimental GPU kernel.
- Ollama, llama.cpp, LM Studio, and self-hosted OpenAI-compatible APIs have
  loopback-only discovery, health checks, model listing, and completion calls.
- Local HTTP is timeout-bounded, limits response size to 16 MiB, and verifies
  resolved loopback addresses before sending traffic.
- The CLI exposes `inspect`, `doctor`, `models`, and `complete` commands.

## P1 — qualification (automated where hardware-independent)

- Strict formatting, Clippy, workspace tests, TypeScript checks, WASM build,
  C ABI smoke test, fuzz smoke, Miri, ASan, dependency policy, coverage, and
  performance checks are present in GitHub Actions.
- Local-stack protocol behavior is covered by an in-process HTTP integration
  test; no installed model or network connection is needed for normal tests.
- The CUDA workflow is deliberately self-hosted and only qualifies Triton/PTX
  after a real NVIDIA runner executes it. It must not be marked complete from
  a machine without that hardware.

## P2 — expansion and future-proofing (implemented)

- Real Vulkan, Metal, and ROCm execution backends with embedded compute kernels
  and resident simulators passing the numerical contract and golden-vector suite.
- Streaming completions via real-time SSE and NDJSON (`--stream`), authenticated
  TLS/LAN support with Bearer token authentication (`--api-key`), and model lifecycle
  management (`pull`, `unload`, `model-info`) as explicit opt-in features; the default
  stays local, loopback, and credential-free.
- Protected release publishing for crates.io and npm artifacts on version tags (`v*`)
  with ownership token safeguards and dry-run verification.
