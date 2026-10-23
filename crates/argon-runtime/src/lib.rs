//! Asynchronous execution graph coordinator, lock-free Single-Producer Single-Consumer (SPSC)
//! ring-buffer allocator, and continuous control scheduling loop for Argon.

#![forbid(non_ascii_idents)]

use argon_kernel::{KernelError, PersistentGridConfig, PlasticityConfig, SramTileSimulator};
use std::cell::UnsafeCell;
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Errors produced by the Argon runtime coordinator and memory subsystem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeError {
    /// Underlying hardware kernel error.
    Kernel(KernelError),
    /// Ring buffer is full; producer cannot enqueue without overwriting unconsumed control frames.
    RingBufferFull {
        /// Capacity of the ring buffer.
        capacity: usize,
    },
    /// Ring buffer is empty; consumer found no ready frames.
    RingBufferEmpty,
    /// Invalid runtime configuration.
    InvalidConfiguration(&'static str),
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Kernel(err) => write!(f, "Runtime kernel error: {err}"),
            Self::RingBufferFull { capacity } => {
                write!(f, "SPSC ring buffer saturated (capacity: {capacity} slots)")
            }
            Self::RingBufferEmpty => write!(f, "SPSC ring buffer empty"),
            Self::InvalidConfiguration(msg) => write!(f, "Invalid runtime config: {msg}"),
        }
    }
}

impl std::error::Error for RuntimeError {}

impl From<KernelError> for RuntimeError {
    fn from(err: KernelError) -> Self {
        Self::Kernel(err)
    }
}

#[repr(align(64))]
#[derive(Debug)]
struct CachePaddedAtomicUsize {
    value: AtomicUsize,
}

impl CachePaddedAtomicUsize {
    const fn new(v: usize) -> Self {
        Self {
            value: AtomicUsize::new(v),
        }
    }
}

/// A single sensory/control telemetry frame stored inside the lock-free ring buffer.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct ControlFrame {
    /// Monotonically increasing sequence identifier.
    pub sequence_id: u64,
    /// Phase polarity (`+1.0` positive real data, `-1.0` negative synthetic/adversarial data).
    pub phase_polarity: f32,
    /// Fixed-dimension sensory feature vector.
    pub features: Vec<f32>,
}

/// Lock-free, bounded Single-Producer Single-Consumer (SPSC) ring buffer.
pub struct SpscRingBuffer {
    capacity: usize,
    mask: usize,
    head: CachePaddedAtomicUsize,
    tail: CachePaddedAtomicUsize,
    slots: Vec<UnsafeCell<Option<ControlFrame>>>,
}

// SAFETY: SpscRingBuffer enforces strict acquire/release memory ordering across independent slots.
unsafe impl Sync for SpscRingBuffer {}
unsafe impl Send for SpscRingBuffer {}

impl SpscRingBuffer {
    /// Creates a new lock-free ring buffer. `capacity` must be a power of two >= 2.
    ///
    /// # Errors
    /// Returns [`RuntimeError::InvalidConfiguration`] if `capacity` is not a power of two or less than 2.
    pub fn with_capacity(capacity: usize) -> Result<Self, RuntimeError> {
        if capacity < 2 || !capacity.is_power_of_two() {
            return Err(RuntimeError::InvalidConfiguration(
                "Ring buffer capacity must be a power of two >= 2",
            ));
        }

        let mut slots = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            slots.push(UnsafeCell::new(None));
        }

        Ok(Self {
            capacity,
            mask: capacity - 1,
            head: CachePaddedAtomicUsize::new(0),
            tail: CachePaddedAtomicUsize::new(0),
            slots,
        })
    }

    /// Enqueues a [`ControlFrame`] without locking.
    ///
    /// # Errors
    /// Returns [`RuntimeError::RingBufferFull`] if all slots are currently occupied.
    pub fn try_push(&self, frame: ControlFrame) -> Result<(), RuntimeError> {
        let head = self.head.value.load(Ordering::Relaxed);
        let tail = self.tail.value.load(Ordering::Acquire);

        if head.wrapping_sub(tail) >= self.capacity {
            return Err(RuntimeError::RingBufferFull {
                capacity: self.capacity,
            });
        }

        let slot_idx = head & self.mask;
        let Some(slot_cell) = self.slots.get(slot_idx) else {
            return Err(RuntimeError::InvalidConfiguration(
                "Slot index out of bounds",
            ));
        };

        // SAFETY: Only producer writes to `head & mask` when `head - tail < capacity`.
        unsafe {
            *slot_cell.get() = Some(frame);
        }

        self.head
            .value
            .store(head.wrapping_add(1), Ordering::Release);
        Ok(())
    }

    /// Dequeues the next available [`ControlFrame`] without locking.
    ///
    /// # Errors
    /// Returns [`RuntimeError::RingBufferEmpty`] if no frames are pending.
    pub fn try_pop(&self) -> Result<ControlFrame, RuntimeError> {
        let tail = self.tail.value.load(Ordering::Relaxed);
        let head = self.head.value.load(Ordering::Acquire);

        if tail == head {
            return Err(RuntimeError::RingBufferEmpty);
        }

        let slot_idx = tail & self.mask;
        let Some(slot_cell) = self.slots.get(slot_idx) else {
            return Err(RuntimeError::InvalidConfiguration(
                "Slot index out of bounds",
            ));
        };

        // SAFETY: Only consumer reads from `tail & mask` when `tail != head`.
        let frame_opt = unsafe { (*slot_cell.get()).take() };

        self.tail
            .value
            .store(tail.wrapping_add(1), Ordering::Release);

        frame_opt.ok_or(RuntimeError::RingBufferEmpty)
    }

    /// Returns the number of frames currently queued in the ring buffer.
    pub fn len(&self) -> usize {
        let head = self.head.value.load(Ordering::Acquire);
        let tail = self.tail.value.load(Ordering::Acquire);
        head.wrapping_sub(tail)
    }

    /// Returns `true` if no frames are queued.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Telemetry emitted after completing a multi-tile Forward-Forward / Hebbian tick.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TickTelemetry {
    /// Frame sequence ID processed.
    pub sequence_id: u64,
    /// Mean squared goodness across all hierarchical SRAM tiles.
    pub mean_goodness: f32,
    /// Final layer normalized activation L2 checksum.
    pub output_l2_norm: f32,
    /// Total cumulative ticks executed by the runtime.
    pub total_ticks: u64,
}

/// Non-blocking multi-layer persistent execution coordinator.
pub struct ArgonCoordinator {
    tile_dim: usize,
    tiles: Vec<SramTileSimulator>,
    scratch_a: Vec<f32>,
    scratch_b: Vec<f32>,
    base_plasticity: PlasticityConfig,
    ring_buffer: SpscRingBuffer,
    total_ticks: u64,
}

impl ArgonCoordinator {
    /// Constructs a hierarchical stack of persistent SRAM tiles.
    ///
    /// # Errors
    /// Returns [`RuntimeError`] if any configuration parameter or tile allocation fails.
    pub fn new(
        num_layers: usize,
        grid_config: &PersistentGridConfig,
        base_plasticity: PlasticityConfig,
        ring_capacity: usize,
    ) -> Result<Self, RuntimeError> {
        if num_layers == 0 {
            return Err(RuntimeError::InvalidConfiguration(
                "num_layers must be at least 1",
            ));
        }
        grid_config.validate()?;
        base_plasticity.validate()?;

        let dim = grid_config.tile_dim;
        let weights_per_tile = dim.saturating_mul(dim);
        let mut tiles = Vec::with_capacity(num_layers);

        for layer_idx in 0..num_layers {
            let mut init_weights = vec![0.02_f32; weights_per_tile];
            for i in 0..dim {
                let diag_idx = i.saturating_mul(dim).saturating_add(i);
                if let Some(w) = init_weights.get_mut(diag_idx) {
                    #[allow(clippy::cast_precision_loss)]
                    let layer_scale = 1.0_f32 / ((layer_idx + 1) as f32).sqrt();
                    *w = 0.5_f32 * layer_scale;
                }
            }
            tiles.push(SramTileSimulator::new(grid_config, &init_weights)?);
        }

        Ok(Self {
            tile_dim: dim,
            tiles,
            scratch_a: vec![0.0_f32; dim],
            scratch_b: vec![0.0_f32; dim],
            base_plasticity,
            ring_buffer: SpscRingBuffer::with_capacity(ring_capacity)?,
            total_ticks: 0,
        })
    }

    /// Submits a sensory frame into the lock-free ingress ring buffer.
    ///
    /// # Errors
    /// Returns [`RuntimeError`] if the feature vector length mismatches `tile_dim` or the ring buffer is full.
    pub fn submit_frame(&self, frame: ControlFrame) -> Result<(), RuntimeError> {
        if frame.features.len() != self.tile_dim {
            return Err(RuntimeError::Kernel(KernelError::DimensionMismatch {
                expected: self.tile_dim,
                actual: frame.features.len(),
            }));
        }
        self.ring_buffer.try_push(frame)
    }

    /// Returns the sensory and action dimension expected by this coordinator.
    pub const fn tile_dim(&self) -> usize {
        self.tile_dim
    }

    /// Drains and executes a single frame from the ingress ring buffer across all resident SRAM tiles.
    ///
    /// # Errors
    /// Returns [`RuntimeError`] if the ring buffer is empty or a kernel step fails.
    pub fn poll_and_step(
        &mut self,
        output_action: &mut [f32],
    ) -> Result<TickTelemetry, RuntimeError> {
        let mut discarded_features = Vec::new();
        self.poll_and_step_recycle(output_action, &mut discarded_features)
    }

    /// Drains and executes one frame, returning its feature allocation to the caller.
    ///
    /// Swapping the consumed frame's feature vector into `recycled_features`
    /// lets real-time producers reuse the same allocation on their next tick.
    ///
    /// # Errors
    /// Returns [`RuntimeError`] if the ring buffer is empty, dimensions do not
    /// match, or a kernel step fails.
    pub fn poll_and_step_recycle(
        &mut self,
        output_action: &mut [f32],
        recycled_features: &mut Vec<f32>,
    ) -> Result<TickTelemetry, RuntimeError> {
        if output_action.len() != self.tile_dim {
            return Err(RuntimeError::Kernel(KernelError::DimensionMismatch {
                expected: self.tile_dim,
                actual: output_action.len(),
            }));
        }

        let mut frame = self.ring_buffer.try_pop()?;
        let mut step_plasticity = self.base_plasticity;
        step_plasticity.phase_polarity = frame.phase_polarity;

        self.scratch_a.copy_from_slice(&frame.features);
        std::mem::swap(recycled_features, &mut frame.features);
        let mut total_goodness = 0.0_f32;

        for (layer_idx, tile) in self.tiles.iter_mut().enumerate() {
            let g = if layer_idx % 2 == 0 {
                tile.step_in_sram(&self.scratch_a, &mut self.scratch_b, &step_plasticity)?
            } else {
                tile.step_in_sram(&self.scratch_b, &mut self.scratch_a, &step_plasticity)?
            };
            total_goodness += g;
        }

        let final_slice = if self.tiles.len() % 2 == 1 {
            &self.scratch_b
        } else {
            &self.scratch_a
        };
        output_action.copy_from_slice(final_slice);

        let mut l2_sq = 0.0_f32;
        for &v in output_action.iter() {
            l2_sq = v.mul_add(v, l2_sq);
        }

        self.total_ticks = self.total_ticks.saturating_add(1);
        #[allow(clippy::cast_precision_loss)]
        let mean_goodness = total_goodness / (self.tiles.len() as f32);

        Ok(TickTelemetry {
            sequence_id: frame.sequence_id,
            mean_goodness,
            output_l2_norm: l2_sq.sqrt(),
            total_ticks: self.total_ticks,
        })
    }

    /// Applies local homeostatic synaptic scaling to every resident tile.
    ///
    /// # Errors
    /// Returns [`RuntimeError`] when the target norm is invalid.
    pub fn apply_homeostasis(&mut self, target_norm: f32) -> Result<(), RuntimeError> {
        for tile in &mut self.tiles {
            tile.scale_weights_to_norm(target_norm)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spsc_ring_and_coordinator_pipeline() -> Result<(), RuntimeError> {
        let grid = PersistentGridConfig {
            grid_dim_x: 2,
            block_dim_x: 64,
            shared_mem_bytes: 16_384,
            tile_dim: 8,
        };
        let plasticity = PlasticityConfig {
            learning_rate: 0.02,
            goodness_threshold: 1.5,
            weight_decay: 0.0005,
            phase_polarity: 1.0,
        };

        let mut coord = ArgonCoordinator::new(2, &grid, plasticity, 16)?;
        coord.submit_frame(ControlFrame {
            sequence_id: 42,
            phase_polarity: 1.0,
            features: vec![0.5_f32; 8],
        })?;

        let mut action_out = vec![0.0_f32; 8];
        let telemetry = coord.poll_and_step(&mut action_out)?;

        assert_eq!(telemetry.sequence_id, 42);
        assert_eq!(telemetry.total_ticks, 1);
        assert!(telemetry.mean_goodness > 0.0);
        Ok(())
    }
}
