//! Bare-metal hardware abstraction layer for the Argon execution engine.
//! Provides persistent PTX assembly generation for NVIDIA SM80+ architectures
//! and a deterministic, cache-line-aligned CPU simulation fallback for ARM64/Termux.

#![forbid(non_ascii_idents)]

use std::fmt;
use std::mem::size_of;

/// Error variants emitted by the Argon hardware abstraction layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelError {
    /// Dimension mismatch between input buffers and SRAM tile geometry.
    DimensionMismatch {
        /// Expected number of elements.
        expected: usize,
        /// Actual number of elements provided.
        actual: usize,
    },
    /// Requested shared memory exceeds hardware SM per-block SRAM ceiling.
    SramCapacityExceeded {
        /// Requested bytes.
        requested_bytes: usize,
        /// Hardware ceiling in bytes.
        max_bytes: usize,
    },
    /// Invalid plasticity hyperparameter configuration.
    InvalidHyperparameter(&'static str),
}

impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "Kernel tensor dimension mismatch: expected {expected}, received {actual}"
                )
            }
            Self::SramCapacityExceeded {
                requested_bytes,
                max_bytes,
            } => {
                write!(
                    f,
                    "SRAM budget exceeded: requested {requested_bytes}B, max allowed is {max_bytes}B"
                )
            }
            Self::InvalidHyperparameter(msg) => {
                write!(f, "Invalid kernel hyperparameter: {msg}")
            }
        }
    }
}

impl std::error::Error for KernelError {}

/// Hardware geometry for persistent thread block execution.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PersistentGridConfig {
    /// Number of persistent thread blocks pinned to SMs.
    pub grid_dim_x: u32,
    /// Number of threads per persistent block (must be a multiple of warp size 32).
    pub block_dim_x: u32,
    /// Static shared memory (SRAM) allocation in bytes per block.
    pub shared_mem_bytes: u32,
    /// Feature dimension of the local recurrent tile.
    pub tile_dim: usize,
}

impl PersistentGridConfig {
    /// Maximum per-block SRAM on Ampere/Ada/Hopper without dynamic opt-in (48 KiB default safe ceiling).
    pub const MAX_SAFE_SRAM_BYTES: u32 = 49_152;

    /// Validates hardware alignment and SRAM budgets.
    ///
    /// # Errors
    /// Returns [`KernelError`] if warp alignment or SRAM limits are violated.
    #[allow(clippy::manual_is_multiple_of)]
    pub const fn validate(&self) -> Result<(), KernelError> {
        if self.block_dim_x == 0 || (self.block_dim_x % 32) != 0 {
            return Err(KernelError::InvalidHyperparameter(
                "block_dim_x must be a non-zero multiple of 32 (warp size)",
            ));
        }
        if self.grid_dim_x == 0 {
            return Err(KernelError::InvalidHyperparameter(
                "grid_dim_x must be greater than zero",
            ));
        }
        if self.shared_mem_bytes > Self::MAX_SAFE_SRAM_BYTES {
            return Err(KernelError::SramCapacityExceeded {
                requested_bytes: self.shared_mem_bytes as usize,
                max_bytes: Self::MAX_SAFE_SRAM_BYTES as usize,
            });
        }
        if self.tile_dim == 0 || (self.tile_dim % 4) != 0 {
            return Err(KernelError::InvalidHyperparameter(
                "tile_dim must be a non-zero multiple of 4 for vectorized load/store",
            ));
        }
        Ok(())
    }
}

/// Hyperparameters governing Forward-Forward and local Hebbian updates inside SRAM.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlasticityConfig {
    /// Local learning rate for synaptic adjustments.
    pub learning_rate: f32,
    /// Goodness threshold separating positive and negative phase dynamics.
    pub goodness_threshold: f32,
    /// Weight decay factor preventing unbounded synaptic growth in SRAM.
    pub weight_decay: f32,
    /// Phase polarity: `+1.0` for positive (real sensory stream), `-1.0` for negative (confabulation).
    pub phase_polarity: f32,
}

impl PlasticityConfig {
    /// Validates that floating-point parameters are finite and within stable bounds.
    ///
    /// # Errors
    /// Returns [`KernelError::InvalidHyperparameter`] if any parameter is NaN, infinite, or out of range.
    pub fn validate(&self) -> Result<(), KernelError> {
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 || self.learning_rate > 1.0
        {
            return Err(KernelError::InvalidHyperparameter(
                "learning_rate must be finite and within (0.0, 1.0]",
            ));
        }
        if !self.goodness_threshold.is_finite() || self.goodness_threshold <= 0.0 {
            return Err(KernelError::InvalidHyperparameter(
                "goodness_threshold must be finite and strictly positive",
            ));
        }
        if !self.weight_decay.is_finite() || self.weight_decay < 0.0 || self.weight_decay >= 1.0 {
            return Err(KernelError::InvalidHyperparameter(
                "weight_decay must be finite and within [0.0, 1.0)",
            ));
        }
        if (self.phase_polarity - 1.0).abs() > f32::EPSILON
            && (self.phase_polarity + 1.0).abs() > f32::EPSILON
        {
            return Err(KernelError::InvalidHyperparameter(
                "phase_polarity must be strictly +1.0 (positive phase) or -1.0 (negative phase)",
            ));
        }
        Ok(())
    }
}

/// Static PTX 8.0 assembly template for SM80+ persistent SRAM execution.
pub const PERSISTENT_PTX_KERNEL_SM80: &str = r"
.version 8.0
.target sm_80
.address_size 64

.visible .entry argon_persistent_ff_hebbian_kernel(
    .param .u64 param_ring_ctrl_ptr,
    .param .u64 param_input_stream_ptr,
    .param .u64 param_weights_hbm_ptr,
    .param .u64 param_output_goodness_ptr,
    .param .f32 param_lr,
    .param .f32 param_threshold,
    .param .f32 param_decay
)
{
    .reg .pred %p<4>;
    .reg .b32  %r<16>;
    .reg .b64  %rd<16>;
    .reg .f32  %f<16>;

    .shared .align 16 .f32 sram_weights[4096];

    ld.param.u64 %rd1, [param_ring_ctrl_ptr];
    ld.param.u64 %rd2, [param_input_stream_ptr];
    ld.param.u64 %rd3, [param_weights_hbm_ptr];
    ld.param.u64 %rd4, [param_output_goodness_ptr];
    ld.param.f32 %f1,  [param_lr];
    ld.param.f32 %f2,  [param_threshold];
    ld.param.f32 %f3,  [param_decay];

    mov.u32 %r1, %tid.x;
    shl.b32 %r2, %r1, 2;
    cvt.u64.u32 %rd5, %r2;

    add.u64 %rd6, %rd3, %rd5;
    ld.global.ca.f32 %f4, [%rd6];
    mov.u32 %r3, sram_weights;
    add.u32 %r4, %r3, %r2;
    st.shared.f32 [%r4], %f4;
    bar.sync 0;

PERSISTENT_SPIN_LOOP:
    ld.global.acquire.gpu.u32 %r5, [%rd1];
    setp.eq.u32 %p1, %r5, 255;
    @%p1 bra KERNEL_EXIT;

    setp.eq.u32 %p2, %r5, 0;
    @%p2 bra PERSISTENT_SPIN_LOOP;

    add.u64 %rd7, %rd2, %rd5;
    ld.global.cv.f32 %f5, [%rd7];
    ld.shared.f32 %f6, [%r4];

    mul.f32 %f7, %f5, %f6;
    max.f32 %f8, %f7, 0f00000000;
    mul.f32 %f9, %f8, %f8;

    sub.f32 %f10, %f9, %f2;
    mul.f32 %f11, %f10, %f5;
    fma.rn.f32 %f12, %f1, %f11, %f6;
    mul.f32 %f13, %f3, %f6;
    sub.f32 %f14, %f12, %f13;

    st.shared.f32 [%r4], %f14;
    bar.sync 0;

    setp.eq.u32 %p3, %r1, 0;
    @%p3 st.global.wb.f32 [%rd4], %f9;
    @%p3 mov.u32 %r6, 0;
    @%p3 st.global.release.gpu.u32 [%rd1], %r6;

    bra PERSISTENT_SPIN_LOOP;

KERNEL_EXIT:
    ld.shared.f32 %f15, [%r4];
    st.global.wb.f32 [%rd6], %f15;
    ret;
}
";

/// Deterministic CPU/ARM64 simulation state mirroring per-SM SRAM persistence.
#[repr(align(64))]
#[derive(Debug, Clone)]
pub struct SramTileSimulator {
    tile_dim: usize,
    sram_weights: Vec<f32>,
    sram_activations: Vec<f32>,
    cumulative_steps: u64,
}

impl SramTileSimulator {
    /// Initializes a cache-line-aligned simulated SRAM tile.
    ///
    /// # Errors
    /// Returns [`KernelError`] if `grid_config` is invalid or `initial_weights` length mismatches `tile_dim * tile_dim`.
    pub fn new(
        grid_config: &PersistentGridConfig,
        initial_weights: &[f32],
    ) -> Result<Self, KernelError> {
        grid_config.validate()?;
        let expected_len = grid_config.tile_dim.saturating_mul(grid_config.tile_dim);
        if initial_weights.len() != expected_len {
            return Err(KernelError::DimensionMismatch {
                expected: expected_len,
                actual: initial_weights.len(),
            });
        }

        let required_bytes = expected_len
            .saturating_add(grid_config.tile_dim)
            .saturating_mul(size_of::<f32>());
        if required_bytes > grid_config.shared_mem_bytes as usize {
            return Err(KernelError::SramCapacityExceeded {
                requested_bytes: required_bytes,
                max_bytes: grid_config.shared_mem_bytes as usize,
            });
        }

        Ok(Self {
            tile_dim: grid_config.tile_dim,
            sram_weights: initial_weights.to_vec(),
            sram_activations: vec![0.0_f32; grid_config.tile_dim],
            cumulative_steps: 0,
        })
    }

    /// Executes a single fused forward + local Forward-Forward / Hebbian plasticity step in simulated SRAM.
    ///
    /// # Errors
    /// Returns [`KernelError`] if `input_slice` or `output_slice` length does not equal `tile_dim`, or if `plasticity` is invalid.
    pub fn step_in_sram(
        &mut self,
        input_slice: &[f32],
        output_slice: &mut [f32],
        plasticity: &PlasticityConfig,
    ) -> Result<f32, KernelError> {
        if input_slice.len() != self.tile_dim {
            return Err(KernelError::DimensionMismatch {
                expected: self.tile_dim,
                actual: input_slice.len(),
            });
        }
        if output_slice.len() != self.tile_dim {
            return Err(KernelError::DimensionMismatch {
                expected: self.tile_dim,
                actual: output_slice.len(),
            });
        }
        plasticity.validate()?;

        let dim = self.tile_dim;
        let mut raw_norm_sq = 0.0_f32;

        for (i, act_slot) in self.sram_activations.iter_mut().enumerate() {
            let row_offset = i.saturating_mul(dim);
            let Some(weight_row) = self.sram_weights.get(row_offset..row_offset + dim) else {
                return Err(KernelError::DimensionMismatch {
                    expected: row_offset + dim,
                    actual: self.sram_weights.len(),
                });
            };

            let mut acc = 0.0_f32;
            for (&w_ij, &x_j) in weight_row.iter().zip(input_slice.iter()) {
                acc = w_ij.mul_add(x_j, acc);
            }
            let relu = acc.max(0.0_f32);
            *act_slot = relu;
            raw_norm_sq = relu.mul_add(relu, raw_norm_sq);
        }

        #[allow(clippy::cast_precision_loss)]
        let inv_dim = 1.0_f32 / (dim as f32);
        let goodness = raw_norm_sq * inv_dim;

        let margin = plasticity.phase_polarity * (plasticity.goodness_threshold - goodness);
        let modulation = 1.0_f32 / (1.0_f32 + (-margin).exp());
        let effective_lr = plasticity.learning_rate * plasticity.phase_polarity * modulation;
        let inv_rms = 1.0_f32 / (goodness.sqrt() + 1e-6_f32);

        for (i, (&y_i, out_slot)) in self
            .sram_activations
            .iter()
            .zip(output_slice.iter_mut())
            .enumerate()
        {
            let row_offset = i.saturating_mul(dim);
            let Some(weight_row) = self.sram_weights.get_mut(row_offset..row_offset + dim) else {
                return Err(KernelError::DimensionMismatch {
                    expected: row_offset + dim,
                    actual: self.sram_weights.len(),
                });
            };

            let post_synaptic_drive = effective_lr * y_i;
            for (w_ij, &x_j) in weight_row.iter_mut().zip(input_slice.iter()) {
                let decay_term = plasticity.weight_decay * (*w_ij);
                let delta = post_synaptic_drive.mul_add(x_j, -decay_term);
                *w_ij += delta;
            }

            *out_slot = y_i * inv_rms;
        }

        self.cumulative_steps = self.cumulative_steps.saturating_add(1);
        Ok(goodness)
    }

    /// Returns an immutable view of the resident SRAM synaptic weights.
    pub fn weights(&self) -> &[f32] {
        &self.sram_weights
    }

    /// Replaces all resident weights without reallocating tile storage.
    ///
    /// # Errors
    /// Returns [`KernelError::DimensionMismatch`] when the snapshot shape does
    /// not match this tile, or [`KernelError::InvalidHyperparameter`] when a
    /// weight is non-finite.
    pub fn restore_weights(&mut self, weights: &[f32]) -> Result<(), KernelError> {
        if weights.len() != self.sram_weights.len() {
            return Err(KernelError::DimensionMismatch {
                expected: self.sram_weights.len(),
                actual: weights.len(),
            });
        }
        if weights.iter().any(|weight| !weight.is_finite()) {
            return Err(KernelError::InvalidHyperparameter(
                "all restored weights must be finite",
            ));
        }
        self.sram_weights.copy_from_slice(weights);
        Ok(())
    }

    /// Returns a deterministic bitwise checksum of the resident weights.
    pub fn weight_checksum(&self) -> u64 {
        let mut checksum = 0xcbf2_9ce4_8422_2325_u64;
        for &weight in &self.sram_weights {
            checksum ^= u64::from(weight.to_bits());
            checksum = checksum.wrapping_mul(0x0000_0100_0000_01b3);
        }
        checksum
    }

    /// Returns the total number of fused steps executed inside this persistent tile.
    pub const fn cumulative_steps(&self) -> u64 {
        self.cumulative_steps
    }

    /// Rescales the resident synaptic weights to a requested L2 norm.
    ///
    /// This is the local synaptic-scaling operation used by runtime
    /// homeostasis. A zero-norm tile is left unchanged because it has no
    /// direction that can be preserved by rescaling.
    ///
    /// # Errors
    /// Returns [`KernelError::InvalidHyperparameter`] when `target_norm` is
    /// not finite or is not strictly positive.
    pub fn scale_weights_to_norm(&mut self, target_norm: f32) -> Result<(), KernelError> {
        if !target_norm.is_finite() || target_norm <= 0.0 {
            return Err(KernelError::InvalidHyperparameter(
                "homeostatic target_norm must be finite and strictly positive",
            ));
        }

        let mut norm_sq = 0.0_f32;
        for &weight in &self.sram_weights {
            norm_sq = weight.mul_add(weight, norm_sq);
        }
        if norm_sq <= f32::EPSILON {
            return Ok(());
        }

        let scale = target_norm / norm_sq.sqrt();
        for weight in &mut self.sram_weights {
            *weight *= scale;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sram_tile_positive_goodness_increase() -> Result<(), KernelError> {
        let grid = PersistentGridConfig {
            grid_dim_x: 1,
            block_dim_x: 64,
            shared_mem_bytes: 16_384,
            tile_dim: 8,
        };
        let weights = vec![0.25_f32; 64];
        let mut sim = SramTileSimulator::new(&grid, &weights)?;

        let input = [1.0_f32, 0.5, 0.8, 0.2, 0.9, 0.4, 0.7, 0.3];
        let mut output = [0.0_f32; 8];
        let plasticity = PlasticityConfig {
            learning_rate: 0.05,
            goodness_threshold: 2.5,
            weight_decay: 0.001,
            phase_polarity: 1.0,
        };

        let g_initial = sim.step_in_sram(&input, &mut output, &plasticity)?;
        let mut g_final = g_initial;
        for _ in 0..10 {
            g_final = sim.step_in_sram(&input, &mut output, &plasticity)?;
        }

        assert!(g_final > g_initial);
        assert_eq!(sim.cumulative_steps(), 11);
        Ok(())
    }
}
