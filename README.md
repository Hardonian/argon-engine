# Argon Engine (`hardonian/argon-engine`)

Low-level, data-efficient persistent SRAM AI execution engine designed for continuous, sample-efficient physical and control paradigms that bypass synchronous PyTorch/CUDA host-dispatch bottlenecks.

## Workspace Layout
- `crates/argon-kernel`: Bare-metal PTX 8.0 persistent kernel emitter and cache-line-aligned (`#[repr(align(64))]`) ARM64/CPU deterministic SRAM simulator.
- `crates/argon-runtime`: Lock-free Single-Producer Single-Consumer (`SpscRingBuffer`) memory allocator and multi-layer Forward-Forward / Hebbian coordinator.
- `crates/argon-ffi`: Panic-safe C-ABI boundary (`cdylib`/`rlib`) for zero-copy native orchestration.
- `crates/argon-wasm`: WebAssembly SIMD128 (`wasm32-unknown-unknown`) linear-memory ring-buffer bridge for edge and browser workers.
- `kernels/triton`: Persistent in-SRAM Triton micro-kernel with deterministic NumPy fallback.
- `bindings/ts`: Strict TypeScript orchestration harness (`noImplicitAny`, `noUncheckedIndexedAccess`, zero-copy `Float32Array` views).
