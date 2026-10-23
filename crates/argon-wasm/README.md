# Argon WASM raw interface

Build with `cargo build -p argon-wasm --target wasm32-unknown-unknown --release`.
The module exports `argon_wasm_create`, `argon_wasm_input_ptr`,
`argon_wasm_output_ptr`, `argon_wasm_step`, `argon_wasm_step_value`, and
`argon_wasm_destroy`.

JavaScript writes sensory values into a `Float32Array` view at the input offset,
calls `argon_wasm_step`, and reads the output view. No serialization occurs.
Any WebAssembly memory growth invalidates all existing typed-array views; callers
must recreate views after `memory.buffer` changes. Arena pointers are opaque and
must be destroyed exactly once. The API is single-threaded unless the embedding
environment supplies shared WebAssembly memory and external synchronization.
