//! Zero-serialization WebAssembly (`wasm32-unknown-unknown`) linear-memory bridge
//! with hardware SIMD128 acceleration when compiled for WASM targets and portable
//! 4-lane vector unrolling on native ARM64 / x86_64 hosts.

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
    pub fn input_mut_ptr(&mut self) -> *mut f32 {
        self.input_buffer.as_mut_ptr()
    }

    /// Returns a const pointer to the output buffer in linear memory for zero-copy JS `Float32Array` reads.
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
    pub fn simd4_dot_product(a: &[f32], b: &[f32]) -> Result<f32, KernelError> {
        if a.len() != b.len() || a.len() % 4 != 0 {
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
