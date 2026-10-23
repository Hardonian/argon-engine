# Architecture

Argon is a persistent streaming execution system. `argon-kernel` owns the
normative local Forward-Forward operation and resident tile state.
`argon-runtime` schedules multiple tiles and transports owned control frames
through a bounded SPSC queue. `argon-medium` turns physical state into normalized
features, decodes differential actions, integrates the plant, and invokes
homeostasis. `argon-ffi` and `argon-wasm` expose synchronous borrowed-buffer
interfaces. `argon-cli` provides reproducible execution, measurement, recording,
and replay.

The real-time path allocates all storage during construction. Medium frames move
their feature vector into the SPSC ring and recover the same allocation after
execution. Snapshots and human-facing reports are explicitly outside the hot
path and may allocate.

The scalar Rust CPU implementation is normative. Accelerated implementations
must pass the numerical contract and golden-vector suite before being reported
as available. Capability checks must never silently select an unavailable GPU.
