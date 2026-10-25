//! Strict C-ABI boundary for zero-copy native (Node-API / Deno FFI / C++) and WASM integrations.
//! Guarantees panic-free FFI boundaries and explicit status code propagation.

#![forbid(non_ascii_idents)]

use argon_kernel::{PersistentGridConfig, PlasticityConfig};
use argon_runtime::{ArgonCoordinator, TickTelemetry};
use std::ffi::c_char;
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

/// Current binary ABI version for native callers.
pub const ARGON_ABI_VERSION: u32 = 1;

/// Returns the binary ABI version implemented by this library.
#[no_mangle]
pub const extern "C" fn argon_engine_abi_version() -> u32 {
    ARGON_ABI_VERSION
}

/// Returns a static null-terminated description for an [`ArgonStatus`].
#[no_mangle]
pub const extern "C" fn argon_status_message(status: ArgonStatus) -> *const c_char {
    let bytes: &'static [u8] = match status {
        ArgonStatus::Ok => b"operation completed successfully\0",
        ArgonStatus::NullPointer => b"a required pointer was null\0",
        ArgonStatus::InvalidConfig => b"configuration or execution was invalid\0",
        ArgonStatus::DimensionMismatch => b"input or output dimension mismatch\0",
        ArgonStatus::RingBufferStateError => b"ring buffer state error\0",
        ArgonStatus::InternalPanic => b"panic intercepted at the FFI boundary\0",
    };
    bytes.as_ptr().cast()
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

        let telemetry = engine
            .coordinator
            .step_borrowed(sequence_id, phase_polarity, input_slice, output_slice)
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

/// Computes the deterministic bitwise checksum of all resident synaptic weights.
///
/// # Safety
/// - `handle` must be a valid pointer allocated by [`argon_engine_create`].
/// - `out_checksum` must be a valid, non-null pointer to a `u64`.
#[no_mangle]
pub unsafe extern "C" fn argon_engine_weight_checksum(
    handle: *const ArgonEngineHandle,
    out_checksum: *mut u64,
) -> ArgonStatus {
    if handle.is_null() || out_checksum.is_null() {
        return ArgonStatus::NullPointer;
    }
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let engine = unsafe { &*handle };
        let checksum = engine.coordinator.weight_checksum();
        unsafe {
            *out_checksum = checksum;
        }
        ArgonStatus::Ok
    }));
    res.unwrap_or(ArgonStatus::InternalPanic)
}

/// Rescales synaptic weights according to local homeostatic target norm.
///
/// # Safety
/// `handle` must be a valid pointer allocated by [`argon_engine_create`].
#[no_mangle]
pub unsafe extern "C" fn argon_engine_scale_weights(
    handle: *mut ArgonEngineHandle,
    target_norm: f32,
) -> ArgonStatus {
    if handle.is_null() {
        return ArgonStatus::NullPointer;
    }
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let engine = unsafe { &mut *handle };
        match engine.coordinator.apply_homeostasis(target_norm) {
            Ok(()) => ArgonStatus::Ok,
            Err(_) => ArgonStatus::InvalidConfig,
        }
    }));
    res.unwrap_or(ArgonStatus::InternalPanic)
}

/// Exports all resident synaptic weights into caller-allocated memory.
///
/// # Safety
/// - `handle` must be a valid pointer allocated by [`argon_engine_create`].
/// - `out_weights` must point to contiguous buffer of at least `max_len` `f32` elements.
/// - `out_written` must be a valid, non-null pointer to `usize`.
#[no_mangle]
pub unsafe extern "C" fn argon_engine_export_weights(
    handle: *const ArgonEngineHandle,
    out_weights: *mut f32,
    max_len: usize,
    out_written: *mut usize,
) -> ArgonStatus {
    if handle.is_null() || out_weights.is_null() || out_written.is_null() {
        return ArgonStatus::NullPointer;
    }
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let engine = unsafe { &*handle };
        let snapshot = engine.coordinator.snapshot();
        let total_len: usize = snapshot.layer_weights.iter().map(Vec::len).sum();
        if max_len < total_len {
            return ArgonStatus::DimensionMismatch;
        }
        let mut offset = 0;
        for layer in &snapshot.layer_weights {
            let dest = unsafe { slice::from_raw_parts_mut(out_weights.add(offset), layer.len()) };
            dest.copy_from_slice(layer);
            offset += layer.len();
        }
        unsafe {
            *out_written = total_len;
        }
        ArgonStatus::Ok
    }));
    res.unwrap_or(ArgonStatus::InternalPanic)
}

/// Restores synaptic weights from caller-provided contiguous memory.
///
/// # Safety
/// - `handle` must be a valid pointer allocated by [`argon_engine_create`].
/// - `in_weights` must point to contiguous slice of `len` valid `f32` values.
#[no_mangle]
pub unsafe extern "C" fn argon_engine_restore_weights(
    handle: *mut ArgonEngineHandle,
    in_weights: *const f32,
    len: usize,
) -> ArgonStatus {
    if handle.is_null() || in_weights.is_null() {
        return ArgonStatus::NullPointer;
    }
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let engine = unsafe { &mut *handle };
        let snapshot = engine.coordinator.snapshot();
        let expected_len: usize = snapshot.layer_weights.iter().map(Vec::len).sum();
        if len != expected_len {
            return ArgonStatus::DimensionMismatch;
        }
        let mut offset = 0;
        let mut new_snapshot = snapshot;
        for layer in &mut new_snapshot.layer_weights {
            let src = unsafe { slice::from_raw_parts(in_weights.add(offset), layer.len()) };
            layer.copy_from_slice(src);
            offset += layer.len();
        }
        match engine.coordinator.restore(&new_snapshot) {
            Ok(()) => ArgonStatus::Ok,
            Err(_) => ArgonStatus::InvalidConfig,
        }
    }));
    res.unwrap_or(ArgonStatus::InternalPanic)
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
        assert_eq!(argon_engine_abi_version(), 1);
        let mut handle: *mut ArgonEngineHandle = ptr::null_mut();
        let status = unsafe { argon_engine_create(2, 8, 0.03, 2.0, 0.001, 8, &raw mut handle) };
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

        let mut checksum = 0_u64;
        let c_status = unsafe { argon_engine_weight_checksum(handle, &raw mut checksum) };
        assert_eq!(c_status, ArgonStatus::Ok);
        assert_ne!(checksum, 0);

        // Export weights (2 layers * 8 * 8 = 128 floats)
        let mut weights_buf = [0.0_f32; 128];
        let mut written = 0_usize;
        let exp_status = unsafe {
            argon_engine_export_weights(
                handle,
                weights_buf.as_mut_ptr(),
                weights_buf.len(),
                &raw mut written,
            )
        };
        assert_eq!(exp_status, ArgonStatus::Ok);
        assert_eq!(written, 128);

        // Scale weights
        let scale_status = unsafe { argon_engine_scale_weights(handle, 2.5) };
        assert_eq!(scale_status, ArgonStatus::Ok);

        // Restore saved weights
        let rest_status =
            unsafe { argon_engine_restore_weights(handle, weights_buf.as_ptr(), written) };
        assert_eq!(rest_status, ArgonStatus::Ok);

        let mut restored_checksum = 0_u64;
        let _ = unsafe { argon_engine_weight_checksum(handle, &raw mut restored_checksum) };
        assert_eq!(restored_checksum, checksum);

        let destroy_status = unsafe { argon_engine_destroy(handle) };
        assert_eq!(destroy_status, ArgonStatus::Ok);
    }
}
