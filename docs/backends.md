# Backend qualification

| Backend | Availability | Qualification |
| --- | --- | --- |
| Rust scalar CPU | Production reference | Workspace tests and deterministic replay |
| WASM scalar/SIMD layout | Supported | WASM build, native parity tests, TypeScript harness |
| NumPy dense reference | Supported | Deterministic Python execution |
| Triton GPU | Experimental | Requires CUDA runner and parity workflow |
| Persistent PTX | Prototype | Not selected by runtime until driver integration and parity pass |

An experimental backend is never advertised as active merely because source is
present. The GPU workflow records device information and fails if CUDA or Triton
is absent on its explicitly labelled runner.
