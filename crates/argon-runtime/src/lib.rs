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
#[derive(Debug)]
pub struct SpscRingBuffer {
    capacity: usize,
    mask: usize,
    head: CachePaddedAtomicUsize,
    tail: CachePaddedAtomicUsize,
    slots: Vec<UnsafeCell<Option<ControlFrame>>>,
}

/// Borrowed producer endpoint for an [`SpscRingBuffer`].
#[derive(Debug, Clone, Copy)]
pub struct SpscProducer<'a> {
    ring: &'a SpscRingBuffer,
}

/// Borrowed consumer endpoint for an [`SpscRingBuffer`].
#[derive(Debug, Clone, Copy)]
pub struct SpscConsumer<'a> {
    ring: &'a SpscRingBuffer,
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
        self.try_push_recover(frame)
            .map_err(|(error, _frame)| error)
    }

    /// Enqueues a frame while returning ownership when the bounded queue is full.
    ///
    /// # Errors
    /// Returns the error and original frame if no queue slot is available.
    pub fn try_push_recover(
        &self,
        frame: ControlFrame,
    ) -> Result<(), (RuntimeError, ControlFrame)> {
        let head = self.head.value.load(Ordering::Relaxed);
        let tail = self.tail.value.load(Ordering::Acquire);

        if head.wrapping_sub(tail) >= self.capacity {
            return Err((
                RuntimeError::RingBufferFull {
                    capacity: self.capacity,
                },
                frame,
            ));
        }

        let slot_idx = head & self.mask;
        let Some(slot_cell) = self.slots.get(slot_idx) else {
            return Err((
                RuntimeError::InvalidConfiguration("Slot index out of bounds"),
                frame,
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

    /// Creates the single-producer and single-consumer endpoint views.
    pub const fn split(&self) -> (SpscProducer<'_>, SpscConsumer<'_>) {
        (SpscProducer { ring: self }, SpscConsumer { ring: self })
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

/// Compact fixed-layout trace record suitable for binary persistence.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TraceRecord {
    /// Frame sequence identifier.
    pub sequence_id: u64,
    /// Mean layer goodness.
    pub mean_goodness: f32,
    /// Final activation L2 norm.
    pub output_l2_norm: f32,
    /// Queue depth observed after the tick.
    pub queue_depth: u32,
    /// Reserved for forward-compatible flags.
    pub flags: u32,
}

/// Preallocated overwrite-oldest telemetry trace.
#[derive(Debug)]
pub struct TraceBuffer {
    records: Vec<TraceRecord>,
    cursor: usize,
    len: usize,
}

impl TraceBuffer {
    /// Allocates a fixed trace capacity.
    ///
    /// # Errors
    /// Returns [`RuntimeError`] when capacity is zero.
    pub fn with_capacity(capacity: usize) -> Result<Self, RuntimeError> {
        if capacity == 0 {
            return Err(RuntimeError::InvalidConfiguration(
                "trace capacity must be greater than zero",
            ));
        }
        let empty = TraceRecord {
            sequence_id: 0,
            mean_goodness: 0.0,
            output_l2_norm: 0.0,
            queue_depth: 0,
            flags: 0,
        };
        Ok(Self {
            records: vec![empty; capacity],
            cursor: 0,
            len: 0,
        })
    }

    /// Records telemetry without allocating, overwriting the oldest slot when full.
    pub fn record(&mut self, telemetry: TickTelemetry, queue_depth: usize) {
        self.records[self.cursor] = TraceRecord {
            sequence_id: telemetry.sequence_id,
            mean_goodness: telemetry.mean_goodness,
            output_l2_norm: telemetry.output_l2_norm,
            queue_depth: u32::try_from(queue_depth).unwrap_or(u32::MAX),
            flags: 0,
        };
        self.cursor = (self.cursor + 1) % self.records.len();
        self.len = self.len.saturating_add(1).min(self.records.len());
    }

    /// Number of valid records retained.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` when no record has been written.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Copies retained records in chronological order into caller storage.
    ///
    /// Returns the number copied; excess older records are omitted when output
    /// storage is smaller than the retained trace.
    pub fn copy_chronological(&self, output: &mut [TraceRecord]) -> usize {
        let count = output.len().min(self.len);
        let oldest = if self.len == self.records.len() {
            self.cursor
        } else {
            0
        };
        for (offset, slot) in output.iter_mut().take(count).enumerate() {
            *slot = self.records[(oldest + offset) % self.records.len()];
        }
        count
    }
}

/// Owned deterministic checkpoint of coordinator learning state.
#[derive(Debug, Clone, PartialEq)]
pub struct CoordinatorSnapshot {
    /// Row-major weights for each layer in execution order.
    pub layer_weights: Vec<Vec<f32>>,
    /// Number of successfully executed ticks.
    pub total_ticks: u64,
}

/// Policy used when coordinator ingress has no free slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackpressurePolicy {
    /// Preserve queued frames and reject the new frame.
    Reject,
    /// Remove the oldest queued frame and enqueue the new frame.
    DropOldest,
}

/// Deterministic repeating positive/negative learning-phase schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhaseSchedule {
    positive_ticks: u32,
    negative_ticks: u32,
    cursor: u32,
}

impl PhaseSchedule {
    /// Creates a repeating schedule with non-zero positive and negative spans.
    ///
    /// # Errors
    /// Returns [`RuntimeError`] if either span is zero or their sum overflows.
    pub const fn new(positive_ticks: u32, negative_ticks: u32) -> Result<Self, RuntimeError> {
        if positive_ticks == 0
            || negative_ticks == 0
            || positive_ticks.checked_add(negative_ticks).is_none()
        {
            return Err(RuntimeError::InvalidConfiguration(
                "phase schedule spans must be non-zero and must not overflow",
            ));
        }
        Ok(Self {
            positive_ticks,
            negative_ticks,
            cursor: 0,
        })
    }

    /// Returns the next phase polarity and advances the schedule.
    pub const fn next_polarity(&mut self) -> f32 {
        let polarity = if self.cursor < self.positive_ticks {
            1.0
        } else {
            -1.0
        };
        let period = self.positive_ticks + self.negative_ticks;
        self.cursor = (self.cursor + 1) % period;
        polarity
    }
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

    /// Submits a frame and preserves its allocation if submission fails.
    ///
    /// # Errors
    /// Returns the runtime error and original frame when validation or enqueue fails.
    pub fn submit_frame_recover(
        &self,
        frame: ControlFrame,
    ) -> Result<(), (RuntimeError, ControlFrame)> {
        if frame.features.len() != self.tile_dim {
            let actual = frame.features.len();
            return Err((
                RuntimeError::Kernel(KernelError::DimensionMismatch {
                    expected: self.tile_dim,
                    actual,
                }),
                frame,
            ));
        }
        self.ring_buffer.try_push_recover(frame)
    }

    /// Submits according to a bounded-queue backpressure policy.
    ///
    /// The returned frame, when present, is the oldest frame displaced by
    /// [`BackpressurePolicy::DropOldest`].
    ///
    /// # Errors
    /// Returns the runtime error and incoming frame when rejection is selected
    /// or dimensions are invalid.
    pub fn submit_with_policy(
        &mut self,
        frame: ControlFrame,
        policy: BackpressurePolicy,
    ) -> Result<Option<ControlFrame>, (RuntimeError, ControlFrame)> {
        match self.submit_frame_recover(frame) {
            Ok(()) => Ok(None),
            Err((RuntimeError::RingBufferFull { .. }, frame))
                if policy == BackpressurePolicy::DropOldest =>
            {
                let displaced = match self.ring_buffer.try_pop() {
                    Ok(displaced) => displaced,
                    Err(error) => return Err((error, frame)),
                };
                self.ring_buffer
                    .try_push_recover(frame)
                    .map(|()| Some(displaced))
            }
            Err(failure) => Err(failure),
        }
    }

    /// Returns the current number of queued ingress frames.
    pub fn queue_depth(&self) -> usize {
        self.ring_buffer.len()
    }

    /// Returns the sensory and action dimension expected by this coordinator.
    pub const fn tile_dim(&self) -> usize {
        self.tile_dim
    }

    /// Returns a deterministic owned checkpoint of all learned state.
    pub fn snapshot(&self) -> CoordinatorSnapshot {
        CoordinatorSnapshot {
            layer_weights: self
                .tiles
                .iter()
                .map(|tile| tile.weights().to_vec())
                .collect(),
            total_ticks: self.total_ticks,
        }
    }

    /// Restores a checkpoint into existing preallocated tile storage.
    ///
    /// # Errors
    /// Returns [`RuntimeError`] when layer count, weight dimensions, or values
    /// do not match the coordinator.
    pub fn restore(&mut self, snapshot: &CoordinatorSnapshot) -> Result<(), RuntimeError> {
        if snapshot.layer_weights.len() != self.tiles.len() {
            return Err(RuntimeError::InvalidConfiguration(
                "snapshot layer count does not match coordinator",
            ));
        }
        for (tile, weights) in self.tiles.iter_mut().zip(&snapshot.layer_weights) {
            tile.restore_weights(weights)?;
        }
        self.total_ticks = snapshot.total_ticks;
        Ok(())
    }

    /// Returns a deterministic checksum across every resident layer.
    pub fn weight_checksum(&self) -> u64 {
        self.tiles
            .iter()
            .fold(0xcbf2_9ce4_8422_2325_u64, |acc, tile| {
                acc.rotate_left(7) ^ tile.weight_checksum()
            })
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
        self.scratch_a.copy_from_slice(&frame.features);
        std::mem::swap(recycled_features, &mut frame.features);
        self.execute_loaded(frame.sequence_id, frame.phase_polarity, output_action)
    }

    /// Executes borrowed features directly without queueing or allocating.
    ///
    /// This is intended for synchronous FFI and WASM callers. Streaming
    /// producers should continue to use [`Self::submit_frame`].
    ///
    /// # Errors
    /// Returns [`RuntimeError`] for invalid dimensions, phase polarity, or
    /// kernel execution failures.
    pub fn step_borrowed(
        &mut self,
        sequence_id: u64,
        phase_polarity: f32,
        features: &[f32],
        output_action: &mut [f32],
    ) -> Result<TickTelemetry, RuntimeError> {
        if features.len() != self.tile_dim {
            return Err(RuntimeError::Kernel(KernelError::DimensionMismatch {
                expected: self.tile_dim,
                actual: features.len(),
            }));
        }
        if output_action.len() != self.tile_dim {
            return Err(RuntimeError::Kernel(KernelError::DimensionMismatch {
                expected: self.tile_dim,
                actual: output_action.len(),
            }));
        }
        self.scratch_a.copy_from_slice(features);
        self.execute_loaded(sequence_id, phase_polarity, output_action)
    }

    /// Drains up to one action vector per telemetry slot into flat output storage.
    ///
    /// An empty ring ends the drain successfully. Other runtime errors are
    /// propagated after already-completed outputs remain valid.
    ///
    /// # Errors
    /// Returns [`RuntimeError`] when output storage has the wrong size or a
    /// kernel operation fails.
    pub fn poll_batch(
        &mut self,
        output_actions: &mut [f32],
        telemetry: &mut [TickTelemetry],
    ) -> Result<usize, RuntimeError> {
        let expected = telemetry.len().saturating_mul(self.tile_dim);
        if output_actions.len() != expected {
            return Err(RuntimeError::Kernel(KernelError::DimensionMismatch {
                expected,
                actual: output_actions.len(),
            }));
        }
        let mut completed = 0_usize;
        for (output, telemetry_slot) in output_actions
            .chunks_exact_mut(self.tile_dim)
            .zip(telemetry.iter_mut())
        {
            match self.poll_and_step(output) {
                Ok(value) => {
                    *telemetry_slot = value;
                    completed = completed.saturating_add(1);
                }
                Err(RuntimeError::RingBufferEmpty) => break,
                Err(error) => return Err(error),
            }
        }
        Ok(completed)
    }

    fn execute_loaded(
        &mut self,
        sequence_id: u64,
        phase_polarity: f32,
        output_action: &mut [f32],
    ) -> Result<TickTelemetry, RuntimeError> {
        let mut step_plasticity = self.base_plasticity;
        step_plasticity.phase_polarity = phase_polarity;
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
            sequence_id,
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

    /// Rescales all resident layer weights towards the target homeostatic norm (alias for [`apply_homeostasis`]).
    ///
    /// # Errors
    /// Returns [`RuntimeError`] when the target norm is invalid.
    pub fn scale_weights(&mut self, target_norm: f32) -> Result<(), RuntimeError> {
        self.apply_homeostasis(target_norm)
    }

    /// Applies an independent homeostatic target norm to each layer.
    ///
    /// # Errors
    /// Returns [`RuntimeError`] when target count differs from layer count or a
    /// target norm is invalid.
    pub fn apply_homeostasis_per_layer(
        &mut self,
        target_norms: &[f32],
    ) -> Result<(), RuntimeError> {
        if target_norms.len() != self.tiles.len() {
            return Err(RuntimeError::InvalidConfiguration(
                "homeostatic target count must match layer count",
            ));
        }
        for (tile, &target_norm) in self.tiles.iter_mut().zip(target_norms) {
            tile.scale_weights_to_norm(target_norm)?;
        }
        Ok(())
    }
}

impl SpscProducer<'_> {
    /// Enqueues a frame, returning it to the caller on failure.
    ///
    /// # Errors
    /// Returns the queue error and unconsumed frame when the queue is full.
    pub fn try_push(&self, frame: ControlFrame) -> Result<(), (RuntimeError, ControlFrame)> {
        self.ring.try_push_recover(frame)
    }
}

impl SpscConsumer<'_> {
    /// Dequeues one ready frame.
    ///
    /// # Errors
    /// Returns [`RuntimeError::RingBufferEmpty`] when no frame is ready.
    pub fn try_pop(&self) -> Result<ControlFrame, RuntimeError> {
        self.ring.try_pop()
    }
}

/// Telemetry summary emitted by [`ArgonMeshCoordinator`] across all mesh nodes.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshTelemetry {
    /// Number of active nodes in the mesh.
    pub active_nodes: usize,
    /// Aggregate mean goodness across all coordinated nodes.
    pub mean_goodness: f32,
    /// Mean L2 norm across all node outputs.
    pub mean_output_norm: f32,
    /// Sequence ID processed during this mesh tick.
    pub sequence_id: u64,
}

/// Multi-tile concurrent coordinator managing an interconnected mesh of [`ArgonCoordinator`] nodes.
pub struct ArgonMeshCoordinator {
    nodes: Vec<ArgonCoordinator>,
    tile_dim: usize,
}

impl ArgonMeshCoordinator {
    /// Constructs a multi-node coordinator mesh with `num_nodes` identical [`ArgonCoordinator`] nodes.
    ///
    /// # Errors
    /// Returns [`RuntimeError`] if `num_nodes == 0` or node construction fails.
    pub fn new(
        num_nodes: usize,
        num_layers: usize,
        grid_config: &PersistentGridConfig,
        base_plasticity: PlasticityConfig,
        ring_capacity: usize,
    ) -> Result<Self, RuntimeError> {
        if num_nodes == 0 {
            return Err(RuntimeError::InvalidConfiguration(
                "mesh node count must be greater than zero",
            ));
        }
        let mut nodes = Vec::with_capacity(num_nodes);
        for _ in 0..num_nodes {
            nodes.push(ArgonCoordinator::new(
                num_layers,
                grid_config,
                base_plasticity,
                ring_capacity,
            )?);
        }
        Ok(Self {
            nodes,
            tile_dim: grid_config.tile_dim,
        })
    }

    /// Executes one synchronous borrowed-buffer step across all mesh nodes in lockstep.
    ///
    /// # Errors
    /// Returns [`RuntimeError`] if input or output slices do not match `num_nodes * tile_dim`.
    pub fn step_mesh(
        &mut self,
        sequence_id: u64,
        phase_polarity: f32,
        input_slices: &[f32],
        output_slices: &mut [f32],
    ) -> Result<MeshTelemetry, RuntimeError> {
        let expected_len = self.nodes.len().saturating_mul(self.tile_dim);
        if input_slices.len() != expected_len {
            return Err(RuntimeError::Kernel(KernelError::DimensionMismatch {
                expected: expected_len,
                actual: input_slices.len(),
            }));
        }
        if output_slices.len() != expected_len {
            return Err(RuntimeError::Kernel(KernelError::DimensionMismatch {
                expected: expected_len,
                actual: output_slices.len(),
            }));
        }

        let mut total_goodness = 0.0_f32;
        let mut total_norm = 0.0_f32;

        for (i, node) in self.nodes.iter_mut().enumerate() {
            let offset = i.saturating_mul(self.tile_dim);
            let in_chunk = &input_slices[offset..offset + self.tile_dim];
            let out_chunk = &mut output_slices[offset..offset + self.tile_dim];
            let telem = node.step_borrowed(sequence_id, phase_polarity, in_chunk, out_chunk)?;
            total_goodness += telem.mean_goodness;
            total_norm += telem.output_l2_norm;
        }

        #[allow(clippy::cast_precision_loss)]
        let inv_n = 1.0_f32 / (self.nodes.len() as f32);
        Ok(MeshTelemetry {
            active_nodes: self.nodes.len(),
            mean_goodness: total_goodness * inv_n,
            mean_output_norm: total_norm * inv_n,
            sequence_id,
        })
    }

    /// Returns the number of nodes in the mesh.
    pub const fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the feature dimension of individual nodes.
    pub const fn tile_dim(&self) -> usize {
        self.tile_dim
    }

    /// Rescales weights uniformly across all nodes in the mesh.
    ///
    /// # Errors
    /// Returns [`RuntimeError`] if any scaling parameter is invalid.
    pub fn scale_mesh_weights(&mut self, target_norm: f32) -> Result<(), RuntimeError> {
        for node in &mut self.nodes {
            node.scale_weights(target_norm)?;
        }
        Ok(())
    }

    /// Computes the aggregate XOR checksum over all nodes in the mesh.
    pub fn mesh_weight_checksum(&self) -> u64 {
        let mut checksum = 0x5a5a_5a5a_5a5a_5a5a_u64;
        for (i, node) in self.nodes.iter().enumerate() {
            let node_sum = node.weight_checksum();
            checksum = checksum.rotate_left(7) ^ node_sum.wrapping_add(i as u64);
        }
        checksum
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

    #[test]
    fn failed_enqueue_returns_feature_allocation() -> Result<(), RuntimeError> {
        let ring = SpscRingBuffer::with_capacity(2)?;
        let (producer, consumer) = ring.split();
        for sequence_id in 0..2 {
            producer
                .try_push(ControlFrame {
                    sequence_id,
                    phase_polarity: 1.0,
                    features: vec![0.5; 4],
                })
                .map_err(|(error, _frame)| error)?;
        }
        let rejected = ControlFrame {
            sequence_id: 2,
            phase_polarity: 1.0,
            features: vec![2.0; 4],
        };
        let pointer = rejected.features.as_ptr();
        let Err((_error, recovered)) = producer.try_push(rejected) else {
            return Err(RuntimeError::InvalidConfiguration(
                "full ring unexpectedly accepted a frame",
            ));
        };
        assert_eq!(pointer, recovered.features.as_ptr());
        assert_eq!(consumer.try_pop()?.sequence_id, 0);
        assert_eq!(consumer.try_pop()?.sequence_id, 1);
        Ok(())
    }

    #[test]
    fn coordinator_snapshot_restores_checksum() -> Result<(), RuntimeError> {
        let grid = PersistentGridConfig {
            grid_dim_x: 1,
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
        let mut coordinator = ArgonCoordinator::new(2, &grid, plasticity, 4)?;
        let snapshot = coordinator.snapshot();
        let initial_checksum = coordinator.weight_checksum();
        coordinator.submit_frame(ControlFrame {
            sequence_id: 0,
            phase_polarity: 1.0,
            features: vec![0.5; 8],
        })?;
        let mut output = [0.0; 8];
        let _telemetry = coordinator.poll_and_step(&mut output)?;
        assert_ne!(initial_checksum, coordinator.weight_checksum());
        coordinator.restore(&snapshot)?;
        assert_eq!(initial_checksum, coordinator.weight_checksum());
        Ok(())
    }

    #[test]
    fn phase_schedule_and_batch_backpressure_are_deterministic() -> Result<(), RuntimeError> {
        let mut phases = PhaseSchedule::new(2, 1)?;
        let actual = [
            phases.next_polarity(),
            phases.next_polarity(),
            phases.next_polarity(),
            phases.next_polarity(),
        ]
        .map(f32::to_bits);
        assert_eq!(actual, [1.0, 1.0, -1.0, 1.0].map(f32::to_bits));
        let grid = PersistentGridConfig {
            grid_dim_x: 1,
            block_dim_x: 64,
            shared_mem_bytes: 16_384,
            tile_dim: 8,
        };
        let plasticity = PlasticityConfig {
            learning_rate: 0.01,
            goodness_threshold: 1.0,
            weight_decay: 0.001,
            phase_polarity: 1.0,
        };
        let mut coordinator = ArgonCoordinator::new(1, &grid, plasticity, 2)?;
        for sequence_id in 0..2 {
            coordinator.submit_frame(ControlFrame {
                sequence_id,
                phase_polarity: 1.0,
                features: vec![0.25; 8],
            })?;
        }
        let displaced = coordinator
            .submit_with_policy(
                ControlFrame {
                    sequence_id: 2,
                    phase_polarity: 1.0,
                    features: vec![0.25; 8],
                },
                BackpressurePolicy::DropOldest,
            )
            .map_err(|(error, _frame)| error)?;
        assert_eq!(displaced.map(|frame| frame.sequence_id), Some(0));
        let mut output = [0.0; 16];
        let empty = TickTelemetry {
            sequence_id: 0,
            mean_goodness: 0.0,
            output_l2_norm: 0.0,
            total_ticks: 0,
        };
        let mut telemetry = [empty; 2];
        assert_eq!(coordinator.poll_batch(&mut output, &mut telemetry)?, 2);
        assert_eq!([telemetry[0].sequence_id, telemetry[1].sequence_id], [1, 2]);
        Ok(())
    }

    #[test]
    fn mesh_coordinator_executes_multi_node_lockstep() -> Result<(), RuntimeError> {
        let grid = PersistentGridConfig {
            grid_dim_x: 1,
            block_dim_x: 64,
            shared_mem_bytes: 16_384,
            tile_dim: 4,
        };
        let plasticity = PlasticityConfig {
            learning_rate: 0.02,
            goodness_threshold: 1.0,
            weight_decay: 0.001,
            phase_polarity: 1.0,
        };
        let mut mesh = ArgonMeshCoordinator::new(3, 2, &grid, plasticity, 4)?;
        assert_eq!(mesh.node_count(), 3);
        assert_eq!(mesh.tile_dim(), 4);

        let initial_checksum = mesh.mesh_weight_checksum();
        assert_ne!(initial_checksum, 0);

        let input_mesh = [
            0.5_f32, 0.4, 0.3, 0.2, // Node 0
            0.1, 0.6, 0.7, 0.8,     // Node 1
            0.9, 0.2, 0.3, 0.4,     // Node 2
        ];
        let mut output_mesh = [0.0_f32; 12];

        let telem = mesh.step_mesh(100, 1.0, &input_mesh, &mut output_mesh)?;
        assert_eq!(telem.active_nodes, 3);
        assert_eq!(telem.sequence_id, 100);
        assert!(telem.mean_goodness > 0.0);
        assert!(telem.mean_output_norm > 0.0);

        let stepped_checksum = mesh.mesh_weight_checksum();
        assert_ne!(stepped_checksum, initial_checksum);

        mesh.scale_mesh_weights(1.5)?;
        let scaled_checksum = mesh.mesh_weight_checksum();
        assert_ne!(scaled_checksum, stepped_checksum);
        Ok(())
    }
}
