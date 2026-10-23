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

## P2 — optional expansion

- Add a real Vulkan/Metal/ROCm execution backend only with an implementation
  that passes the existing numerical contract and golden-vector suite.
- Add streaming completions, authenticated TLS/LAN support, or model lifecycle
  management only as explicit opt-in features; the default stays local and
  credential-free.
- Publish crates/npm artifacts only from a protected release tag with the
  required registry ownership credentials.

The P2 entries intentionally remain conditional: claiming production support
for an unimplemented accelerator or an unverified external service would make
the capability report unreliable.
