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
- `crates/argon-backend`: Explicit backend abstraction and non-selecting CUDA capability discovery.
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
