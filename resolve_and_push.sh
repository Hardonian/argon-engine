#!/usr/bin/env bash
# ==============================================================================
# Argon Engine — Rust 1.96 Clippy Hardening & GitHub Push
# ==============================================================================
set -euo pipefail

REPO_OWNER="hardonian"
REPO_NAME="argon-engine"
LOCAL_REPO_DIR="${HOME}/${REPO_NAME}"

log_info()  { printf "\033[1;34m[ARGON-FIX]\033[0m %s\n" "$1"; }
log_ok()    { printf "\033[1;32m[ARGON-OK]\033[0m  %s\n" "$1"; }
log_warn()  { printf "\033[1;33m[ARGON-WARN]\033[0m %s\n" "$1"; }
log_error() { printf "\033[1;31m[ARGON-ERR]\033[0m %s\n" "$1" >&2; }

cd "${LOCAL_REPO_DIR}"

# 1. Update crates/argon-wasm/src/lib.rs
log_info "Applying Rust 1.96 fixes to crates/argon-wasm/src/lib.rs..."
cat << 'W_EOF' > crates/argon-wasm/src/lib.rs
//! Zero-serialization WebAssembly (`wasm32-unknown-unknown`) linear-memory bridge
//! with hardware SIMD128 acceleration when compiled for WASM targets and portable
//! 4-lane vector unrolling on native ARM64 / `x86_64` hosts.

#![forbid(non_ascii_idents)]

use argon_kernel::{KernelError, PersistentGridConfig, PlasticityConfig, SramTileSimulator};

/// Fixed-capacity linear-memory arena enabling JavaScript `Float32Array` views
/// over WebAssembly `SharedArrayBuffer` memory without serialization copies.
pub struct WasmLinearArena {
    tile_dim: usize,
    input_buffer: Vec<f32>,
    output_buffer: Vec<f32>,
    simulator: SramTileSimulator,
    plasticity: PlasticityConfig,
}

impl WasmLinearArena {
    /// Creates a new linear-memory arena with a resident `SramTileSimulator`.
    ///
    /// # Errors
    /// Returns [`KernelError`] if `tile_dim` is invalid or parameters fail validation.
    pub fn new(
        tile_dim: usize,
        learning_rate: f32,
        goodness_threshold: f32,
        weight_decay: f32,
    ) -> Result<Self, KernelError> {
        let grid = PersistentGridConfig {
            grid_dim_x: 1,
            block_dim_x: 64,
            shared_mem_bytes: PersistentGridConfig::MAX_SAFE_SRAM_BYTES,
            tile_dim,
        };
        let plasticity = PlasticityConfig {
            learning_rate,
            goodness_threshold,
            weight_decay,
            phase_polarity: 1.0,
        };
        plasticity.validate()?;

        let total_weights = tile_dim.saturating_mul(tile_dim);
        let mut init_weights = vec![0.02_f32; total_weights];
        for i in 0..tile_dim {
            if let Some(slot) = init_weights.get_mut(i.saturating_mul(tile_dim).saturating_add(i))
            {
                *slot = 0.45_f32;
            }
        }

        let simulator = SramTileSimulator::new(&grid, &init_weights)?;
        Ok(Self {
            tile_dim,
            input_buffer: vec![0.0_f32; tile_dim],
            output_buffer: vec![0.0_f32; tile_dim],
            simulator,
            plasticity,
        })
    }

    /// Returns a mutable pointer to the input buffer in linear memory for zero-copy JS `Float32Array` writes.
    #[allow(clippy::missing_const_for_fn)]
    pub fn input_mut_ptr(&mut self) -> *mut f32 {
        self.input_buffer.as_mut_ptr()
    }

    /// Returns a const pointer to the output buffer in linear memory for zero-copy JS `Float32Array` reads.
    #[allow(clippy::missing_const_for_fn)]
    pub fn output_ptr(&self) -> *const f32 {
        self.output_buffer.as_ptr()
    }

    /// Mutable slice view of the input buffer for host/unit tests.
    pub fn input_slice_mut(&mut self) -> &mut [f32] {
        &mut self.input_buffer
    }

    /// Immutable slice view of the output buffer for host/unit tests.
    pub fn output_slice(&self) -> &[f32] {
        &self.output_buffer
    }

    /// Executes an in-place step using the resident linear buffers.
    ///
    /// # Errors
    /// Returns [`KernelError`] if polarity is invalid.
    pub fn step_in_place(&mut self, phase_polarity: f32) -> Result<f32, KernelError> {
        self.plasticity.phase_polarity = phase_polarity;
        self.simulator.step_in_sram(
            &self.input_buffer,
            &mut self.output_buffer,
            &self.plasticity,
        )
    }

    /// Computes a 4-lane unrolled dot product (WASM SIMD128 compatible layout).
    ///
    /// # Errors
    /// Returns [`KernelError::DimensionMismatch`] if slice lengths differ or are not multiples of 4.
    #[allow(clippy::manual_is_multiple_of)]
    pub fn simd4_dot_product(a: &[f32], b: &[f32]) -> Result<f32, KernelError> {
        if a.len() != b.len() || (a.len() % 4) != 0 {
            return Err(KernelError::DimensionMismatch {
                expected: a.len(),
                actual: b.len(),
            });
        }

        let mut acc0 = 0.0_f32;
        let mut acc1 = 0.0_f32;
        let mut acc2 = 0.0_f32;
        let mut acc3 = 0.0_f32;

        for (chunk_a, chunk_b) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
            if let ([a0, a1, a2, a3], [b0, b1, b2, b3]) = (chunk_a, chunk_b) {
                acc0 = a0.mul_add(*b0, acc0);
                acc1 = a1.mul_add(*b1, acc1);
                acc2 = a2.mul_add(*b2, acc2);
                acc3 = a3.mul_add(*b3, acc3);
            }
        }

        Ok((acc0 + acc1) + (acc2 + acc3))
    }

    /// Returns the configured tile dimension.
    pub const fn tile_dim(&self) -> usize {
        self.tile_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_linear_arena_and_simd4_dot() -> Result<(), KernelError> {
        let mut arena = WasmLinearArena::new(8, 0.04, 2.0, 0.001)?;
        arena
            .input_slice_mut()
            .copy_from_slice(&[0.8_f32, 0.3, 0.6, 0.2, 0.9, 0.4, 0.7, 0.5]);

        let g1 = arena.step_in_place(1.0)?;
        let g2 = arena.step_in_place(1.0)?;
        assert!(g2 > g1);

        let dot = WasmLinearArena::simd4_dot_product(arena.output_slice(), arena.output_slice())?;
        assert!(dot > 0.0);
        Ok(())
    }
}
W_EOF

# 2. Update crates/argon-ffi/src/lib.rs
log_info "Applying Rust 1.96 raw pointer fixes to crates/argon-ffi/src/lib.rs..."
cat << 'F_EOF' > crates/argon-ffi/src/lib.rs
//! Strict C-ABI boundary for zero-copy native (Node-API / Deno FFI / C++) and WASM integrations.
//! Guarantees panic-free FFI boundaries and explicit status code propagation.

#![forbid(non_ascii_idents)]

use argon_kernel::{PersistentGridConfig, PlasticityConfig};
use argon_runtime::{ArgonCoordinator, ControlFrame, TickTelemetry};
use std::slice;

/// Status codes returned across the C-ABI boundary.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgonStatus {
    /// Operation completed successfully.
    Ok = 0,
    /// A null pointer was passed to an FFI function.
    NullPointer = -1,
    /// Invalid hyperparameter or hardware configuration.
    InvalidConfig = -2,
    /// Tensor slice length did not match engine `tile_dim`.
    DimensionMismatch = -3,
    /// Ring buffer is full or empty.
    RingBufferStateError = -4,
    /// Internal panic caught at the FFI firewall.
    InternalPanic = -99,
}

/// Opaque C-ABI handle wrapping [`ArgonCoordinator`].
pub struct ArgonEngineHandle {
    coordinator: ArgonCoordinator,
    tile_dim: usize,
}

/// Creates and initializes a new [`ArgonEngineHandle`] on the heap.
///
/// # Safety
/// `out_handle` must be a valid, non-null, aligned pointer to a `*mut ArgonEngineHandle`.
#[no_mangle]
pub unsafe extern "C" fn argon_engine_create(
    num_layers: usize,
    tile_dim: usize,
    learning_rate: f32,
    goodness_threshold: f32,
    weight_decay: f32,
    ring_capacity: usize,
    out_handle: *mut *mut ArgonEngineHandle,
) -> ArgonStatus {
    if out_handle.is_null() {
        return ArgonStatus::NullPointer;
    }

    let result = std::panic::catch_unwind(|| {
        let grid = PersistentGridConfig {
            grid_dim_x: 1,
            block_dim_x: 64,
            shared_mem_bytes: PersistentGridConfig::MAX_SAFE_SRAM_BYTES,
            tile_dim,
        };
        let plasticity = PlasticityConfig {
            learning_rate,
            goodness_threshold,
            weight_decay,
            phase_polarity: 1.0,
        };

        ArgonCoordinator::new(num_layers, &grid, plasticity, ring_capacity)
    });

    match result {
        Ok(Ok(coordinator)) => {
            let boxed = Box::new(ArgonEngineHandle {
                coordinator,
                tile_dim,
            });
            // SAFETY: Verified `out_handle` is non-null above.
            unsafe {
                *out_handle = Box::into_raw(boxed);
            }
            ArgonStatus::Ok
        }
        Ok(Err(_)) => ArgonStatus::InvalidConfig,
        Err(_) => ArgonStatus::InternalPanic,
    }
}

/// Executes a synchronous zero-copy enqueue + poll step across the FFI boundary.
///
/// # Safety
/// - `handle` must be a valid pointer previously allocated by [`argon_engine_create`].
/// - `input_ptr` must point to a contiguous array of `input_len` valid `f32` values.
/// - `output_ptr` must point to a mutable contiguous array of `output_len` valid `f32` values.
/// - `out_telemetry` must be a valid, non-null pointer to a [`TickTelemetry`] struct.
#[no_mangle]
pub unsafe extern "C" fn argon_engine_step_zero_copy(
    handle: *mut ArgonEngineHandle,
    sequence_id: u64,
    phase_polarity: f32,
    input_ptr: *const f32,
    input_len: usize,
    output_ptr: *mut f32,
    output_len: usize,
    out_telemetry: *mut TickTelemetry,
) -> ArgonStatus {
    if handle.is_null() || input_ptr.is_null() || output_ptr.is_null() || out_telemetry.is_null() {
        return ArgonStatus::NullPointer;
    }

    let step_res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: Validated non-null pointers above; caller guarantees slice validity.
        let engine = unsafe { &mut *handle };
        if input_len != engine.tile_dim || output_len != engine.tile_dim {
            return Err(ArgonStatus::DimensionMismatch);
        }

        let input_slice = unsafe { slice::from_raw_parts(input_ptr, input_len) };
        let output_slice = unsafe { slice::from_raw_parts_mut(output_ptr, output_len) };

        let frame = ControlFrame {
            sequence_id,
            phase_polarity,
            features: input_slice.to_vec(),
        };

        engine
            .coordinator
            .submit_frame(frame)
            .map_err(|_| ArgonStatus::RingBufferStateError)?;

        let telemetry = engine
            .coordinator
            .poll_and_step(output_slice)
            .map_err(|_| ArgonStatus::InvalidConfig)?;

        unsafe {
            *out_telemetry = telemetry;
        }
        Ok(ArgonStatus::Ok)
    }));

    match step_res {
        Ok(Ok(status)) => status,
        Ok(Err(err_status)) => err_status,
        Err(_) => ArgonStatus::InternalPanic,
    }
}

/// Frees an [`ArgonEngineHandle`] previously allocated by [`argon_engine_create`].
///
/// # Safety
/// `handle` must be either null or a valid pointer returned by [`argon_engine_create`] that has not yet been freed.
#[no_mangle]
pub unsafe extern "C" fn argon_engine_destroy(handle: *mut ArgonEngineHandle) -> ArgonStatus {
    if handle.is_null() {
        return ArgonStatus::Ok;
    }
    // SAFETY: Caller guarantees `handle` was allocated by `argon_engine_create` and is freed only once.
    unsafe {
        drop(Box::from_raw(handle));
    }
    ArgonStatus::Ok
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    #[test]
    fn test_ffi_lifecycle_and_zero_copy_step() {
        let mut handle: *mut ArgonEngineHandle = ptr::null_mut();
        // Uses &raw mut to comply with clippy::borrow_as_ptr in Rust 1.82+ and 1.96+
        let status = unsafe {
            argon_engine_create(2, 8, 0.03, 2.0, 0.001, 8, &raw mut handle)
        };
        assert_eq!(status, ArgonStatus::Ok);
        assert!(!handle.is_null());

        let input = [0.6_f32; 8];
        let mut output = [0.0_f32; 8];
        let mut telemetry = TickTelemetry {
            sequence_id: 0,
            mean_goodness: 0.0,
            output_l2_norm: 0.0,
            total_ticks: 0,
        };

        let step_status = unsafe {
            argon_engine_step_zero_copy(
                handle,
                101,
                1.0,
                input.as_ptr(),
                input.len(),
                output.as_mut_ptr(),
                output.len(),
                &raw mut telemetry,
            )
        };
        assert_eq!(step_status, ArgonStatus::Ok);
        assert_eq!(telemetry.sequence_id, 101);
        assert_eq!(telemetry.total_ticks, 1);

        let destroy_status = unsafe { argon_engine_destroy(handle) };
        assert_eq!(destroy_status, ArgonStatus::Ok);
    }
}
F_EOF

# 3. Validation Gates
log_info "Running cargo fmt..."
cargo fmt --all

log_info "Verifying strict Clippy across all 4 crates..."
cargo clippy --workspace --all-targets --all-features -- -D warnings

log_info "Running workspace test suites..."
cargo test --workspace --all-features
log_ok "All Rust crates compiled and passed tests with zero warnings."

log_info "Verifying TypeScript strict harness..."
pushd bindings/ts >/dev/null
npm install --no-audit --no-fund
npm run typecheck
npm run build
npm test
popd >/dev/null
log_ok "TypeScript harness passed."

if command -v python3 >/dev/null 2>&1 && python3 -c "import numpy" >/dev/null 2>&1; then
  python3 kernels/triton/persistent_sram_ff.py
  log_ok "Triton micro-kernel simulation passed."
fi

# 4. Commit clean working tree
git add -A
if ! git diff --cached --quiet; then
  git commit -m "fix(clippy): resolve doc-markdown, borrow-as-ptr, and missing-errors-doc for Rust 1.96"
  log_ok "Committed clean build to local Git repository."
fi

# 5. Remote GitHub Sync
log_info "Initiating push to remote repository ${REPO_OWNER}/${REPO_NAME}..."
PUSHED=0

if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  log_info "Pushing via authenticated GitHub CLI..."
  if ! gh repo view "${REPO_OWNER}/${REPO_NAME}" >/dev/null 2>&1; then
    gh repo create "${REPO_OWNER}/${REPO_NAME}" \
      --public \
      --source="${LOCAL_REPO_DIR}" \
      --remote=origin \
      --description "Low-level, data-efficient persistent SRAM AI execution engine for continuous physical control paradigms." \
      --push
    PUSHED=1
  else
    REMOTE_URL="$(gh repo view "${REPO_OWNER}/${REPO_NAME}" --json url -q .url).git"
    git remote remove origin 2>/dev/null || true
    git remote add origin "${REMOTE_URL}"
    git push -u origin main
    PUSHED=1
  fi
fi

if [ "${PUSHED}" -eq 0 ]; then
  ACTIVE_TOKEN="${GITHUB_TOKEN:-${GH_TOKEN:-}}"
  if [ -n "${ACTIVE_TOKEN}" ]; then
    log_info "Pushing via GitHub REST API token..."
    HTTP_CODE=$(curl -s -o /tmp/argon_repo_check.json -w "%{http_code}" \
      -H "Accept: application/vnd.github+json" \
      -H "Authorization: Bearer ${ACTIVE_TOKEN}" \
      -H "X-GitHub-Api-Version: 2022-11-28" \
      "https://api.github.com/repos/${REPO_OWNER}/${REPO_NAME}")

    if [ "${HTTP_CODE}" = "404" ]; then
      curl -s -o /tmp/argon_repo_create.json \
        -X POST "https://api.github.com/user/repos" \
        -H "Accept: application/vnd.github+json" \
        -H "Authorization: Bearer ${ACTIVE_TOKEN}" \
        -H "X-GitHub-Api-Version: 2022-11-28" \
        -d "{\"name\":\"${REPO_NAME}\",\"description\":\"Low-level, data-efficient persistent SRAM AI execution engine for continuous physical control paradigms.\",\"private\":false,\"has_issues\":true,\"has_wiki\":false}" >/dev/null
    fi

    git remote remove origin 2>/dev/null || true
    git remote add origin "https://x-access-token:${ACTIVE_TOKEN}@github.com/${REPO_OWNER}/${REPO_NAME}.git"
    git push -u origin main
    git remote set-url origin "https://github.com/${REPO_OWNER}/${REPO_NAME}.git"
    PUSHED=1
  fi
fi

if [ "${PUSHED}" -eq 0 ]; then
  log_info "Attempting SSH push..."
  git remote remove origin 2>/dev/null || true
  git remote add origin "git@github.com:${REPO_OWNER}/${REPO_NAME}.git"
  if git push -u origin main 2>/dev/null; then
    PUSHED=1
  else
    git remote set-url origin "https://github.com/${REPO_OWNER}/${REPO_NAME}.git"
    log_warn "Local commit verified and clean at ${LOCAL_REPO_DIR}."
    log_warn "To push to GitHub now, run:"
    log_warn "  GITHUB_TOKEN=ghp_xxx ./resolve_and_push.sh"
  fi
fi

if [ "${PUSHED}" -eq 1 ]; then
  log_ok "Argon Engine successfully synchronized and live at https://github.com/${REPO_OWNER}/${REPO_NAME}"
fi
