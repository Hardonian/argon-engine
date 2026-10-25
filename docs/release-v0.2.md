# Argon v0.2 release gate

The release candidate is acceptable when all repository-controlled gates below
pass on the tagged commit:

- Formatting, strict Clippy, workspace tests, documentation, and MSRV.
- Miri, Loom, fuzz smoke, AddressSanitizer, dependency policy, and audit.
- Zero allocations over 1,000 warmed closed-loop ticks.
- Conservative energy error below `1e-4` and 10,000-step shock stabilization.
- CPU, WASM, NumPy, and available GPU golden-vector parity.
- Linux, Windows, macOS, WASM, and self-hosted CUDA workflows.
- Reproducible CLI recording/replay and the checked-in performance budget.
- Native header smoke test, npm package dry run, and release artifacts.

## Verification Status

- [x] Workspace tests, strict Clippy, and formatting pass across all crates.
- [x] Zero allocations over warmed closed-loop execution.
- [x] Numerical contract and energy conservation verified.
- [x] Golden-vector parity confirmed across CPU, Vulkan, Metal, ROCm, WASM, and Python NumPy.
- [x] C FFI smoke test (`examples/c_ffi_smoke.c`) passes.
- [x] Python test suite (`bindings/python/tests/test_engine.py`) passes.
- [x] TypeScript test suite (`bindings/ts`) passes.
- [x] Heterogeneous parity CLI command (`argon-cli verify-parity`) passing bit-exact.

Registry publication additionally requires crate/npm ownership tokens. CUDA
qualification requires the `self-hosted, linux, x64, cuda` runner. Those are
external release resources and are intentionally not replaced by simulated
success.
