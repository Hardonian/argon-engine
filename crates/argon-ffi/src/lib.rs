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
        let status = unsafe { argon_engine_create(2, 8, 0.03, 2.0, 0.001, 8, &mut handle) };
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
                &mut telemetry,
            )
        };
        assert_eq!(step_status, ArgonStatus::Ok);
        assert_eq!(telemetry.sequence_id, 101);
        assert_eq!(telemetry.total_ticks, 1);

        let destroy_status = unsafe { argon_engine_destroy(handle) };
        assert_eq!(destroy_status, ArgonStatus::Ok);
    }
}
