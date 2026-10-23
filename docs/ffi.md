# Native embedding

Include `include/argon.h`, link the `argon_ffi` dynamic or static library, and
check `argon_engine_abi_version()` before creating an engine. Handles are opaque,
owned by the caller, and destroyed exactly once. Input and output arrays must
remain valid for the duration of a call and must contain `tile_dim` floats.

Every exported function catches Rust panics. `argon_status_message` returns a
process-lifetime static string and must not be freed. The synchronous step reads
borrowed input directly into preallocated coordinator scratch storage and writes
into caller-owned output storage.
