# Argon Engine (`hardonian/argon-engine`)

Low-level, data-efficient persistent SRAM AI execution engine designed for continuous, sample-efficient physical and control paradigms that bypass synchronous PyTorch/CUDA host-dispatch bottlenecks.

## Workspace Layout
- `crates/argon-kernel`: Bare-metal PTX 8.0 persistent kernel emitter and cache-line-aligned (`#[repr(align(64))]`) ARM64/CPU deterministic SRAM simulator.
- `crates/argon-runtime`: Lock-free Single-Producer Single-Consumer (`SpscRingBuffer`) memory allocator and multi-layer Forward-Forward / Hebbian coordinator.
- `crates/argon-medium`: Cache-line-aligned coupled-oscillator plant with allocation-free symplectic integration and sensory-motor coordinator coupling.
- `crates/argon-ffi`: Panic-safe C-ABI boundary (`cdylib`/`rlib`) for zero-copy native orchestration.
- `crates/argon-wasm`: WebAssembly SIMD128 (`wasm32-unknown-unknown`) linear-memory ring-buffer bridge for edge and browser workers.
- `crates/argon-cli`: Nanosecond latency benchmark harness for coordinator-only and closed-loop physical-medium execution.
- `crates/argon-io`: Allocation-free physical-device adapter contract and deterministic hardware-in-the-loop emulator.
- `crates/argon-backend`: Explicit CPU backend, consumer-GPU inventory, and local-only Ollama, llama.cpp, and OpenAI-compatible model-server client.
- `kernels/triton`: Persistent in-SRAM Triton micro-kernel with deterministic NumPy fallback.
- `bindings/ts`: Strict TypeScript orchestration harness (`noImplicitAny`, `noUncheckedIndexedAccess`, zero-copy `Float32Array` views).

## Quality gates

```text
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo build -p argon-wasm --target wasm32-unknown-unknown --release
python kernels/triton/persistent_sram_ff.py
python scripts/check_benchmark.py
```

Architecture, execution semantics, numerical tolerances, backend qualification,
and native embedding are documented under `docs/`. CUDA qualification runs only
on the explicitly labelled self-hosted GPU workflow and is never inferred from
the presence of kernel source.

## Local-first deployment

Argon is safe to deploy beside a sovereign local model stack: it does not use
cloud credentials or silently route prompts to a remote service. The built-in
client only accepts loopback `http://` endpoints by default and supports
Ollama (`127.0.0.1:11434`), llama.cpp (`127.0.0.1:8080`), LM Studio
(`127.0.0.1:1234`), and a self-hosted OpenAI-compatible server
(`127.0.0.1:8000`).

```text
cargo run -p argon-cli -- inspect --format text
cargo run -p argon-cli -- doctor --local-stack ollama
cargo run -p argon-cli -- doctor --local-stack llama-cpp --endpoint http://127.0.0.1:8080
cargo run -p argon-cli -- models --local-stack lm-studio
cargo run -p argon-cli -- complete --local-stack ollama --model qwen3:8b --prompt "Summarize this control trace" --stream
cargo run -p argon-cli -- pull --local-stack ollama --model qwen3:8b
cargo run -p argon-cli -- unload --local-stack ollama --model qwen3:8b
cargo run -p argon-cli -- model-info --local-stack ollama --model qwen3:8b
```

`inspect` reports visible NVIDIA VRAM, accelerated platforms (Vulkan, Metal, ROCm), and conservative local-model guidance;
`doctor` verifies that the selected server responds; `models` lists available
models; `complete` sends a prompt to its native local API (supporting `--stream`);
`pull`, `unload`, and `model-info` manage model lifecycles; and `--api-key` enables
authenticated endpoints. Remote or LAN endpoints are rejected unless
`--allow-non-loopback` is deliberately supplied.
