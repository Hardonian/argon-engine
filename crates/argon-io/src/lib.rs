//! Hardware-in-the-loop transport traits and a deterministic emulated device.

#![forbid(non_ascii_idents)]

use argon_runtime::{ArgonCoordinator, RuntimeError, TickTelemetry};
use std::fmt;

/// Errors produced by a physical or emulated device adapter.
#[derive(Debug)]
pub enum IoError {
    /// Device and coordinator dimensions are incompatible.
    DimensionMismatch,
    /// The device transport failed.
    Transport(&'static str),
    /// Neural execution failed.
    Runtime(RuntimeError),
}

impl fmt::Display for IoError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch => write!(formatter, "device dimensions are incompatible"),
            Self::Transport(message) => write!(formatter, "device transport failed: {message}"),
            Self::Runtime(error) => write!(formatter, "device runtime failed: {error}"),
        }
    }
}

impl std::error::Error for IoError {}

impl From<RuntimeError> for IoError {
    fn from(error: RuntimeError) -> Self {
        Self::Runtime(error)
    }
}

/// Synchronous fixed-dimension physical device contract.
pub trait PhysicalDevice {
    /// Number of sensory and action channels.
    fn dimension(&self) -> usize;

    /// Writes one current sensor sample into caller-owned storage.
    ///
    /// # Errors
    /// Returns [`IoError`] when the transport or output buffer is invalid.
    fn read_sensors(&mut self, output: &mut [f32]) -> Result<(), IoError>;

    /// Applies one caller-owned actuator vector.
    ///
    /// # Errors
    /// Returns [`IoError`] when the transport or input buffer is invalid.
    fn write_actuators(&mut self, actions: &[f32]) -> Result<(), IoError>;
}

/// Preallocated neural loop around a physical device adapter.
#[derive(Debug)]
pub struct DeviceLoop<D> {
    device: D,
    sensors: Vec<f32>,
    actions: Vec<f32>,
    sequence_id: u64,
}

impl<D: PhysicalDevice> DeviceLoop<D> {
    /// Creates a loop after checking device/coordinator dimensions.
    ///
    /// # Errors
    /// Returns [`IoError::DimensionMismatch`] for incompatible dimensions.
    pub fn new(device: D, coordinator: &ArgonCoordinator) -> Result<Self, IoError> {
        let dimension = device.dimension();
        if dimension == 0 || dimension != coordinator.tile_dim() {
            return Err(IoError::DimensionMismatch);
        }
        Ok(Self {
            device,
            sensors: vec![0.0; dimension],
            actions: vec![0.0; dimension],
            sequence_id: 0,
        })
    }

    /// Executes one allocation-free hardware sensory-control tick.
    ///
    /// # Errors
    /// Returns [`IoError`] for device or neural failures.
    pub fn tick(&mut self, coordinator: &mut ArgonCoordinator) -> Result<TickTelemetry, IoError> {
        self.device.read_sensors(&mut self.sensors)?;
        let telemetry =
            coordinator.step_borrowed(self.sequence_id, 1.0, &self.sensors, &mut self.actions)?;
        self.device.write_actuators(&self.actions)?;
        self.sequence_id = self.sequence_id.wrapping_add(1);
        Ok(telemetry)
    }

    /// Returns the underlying adapter.
    pub const fn device(&self) -> &D {
        &self.device
    }
}

/// Deterministic first-order emulated device for CI and integration development.
#[derive(Debug, Clone)]
pub struct LoopbackDevice {
    state: Vec<f32>,
    response: f32,
}

impl LoopbackDevice {
    /// Creates an emulated device with a response coefficient in `(0, 1]`.
    ///
    /// # Errors
    /// Returns [`IoError`] when dimensions or response are invalid.
    pub fn new(dimension: usize, response: f32) -> Result<Self, IoError> {
        if dimension == 0 || !response.is_finite() || response <= 0.0 || response > 1.0 {
            return Err(IoError::Transport(
                "loopback dimension and response are invalid",
            ));
        }
        Ok(Self {
            state: vec![0.25; dimension],
            response,
        })
    }

    /// Returns the current emulated sensor state.
    pub fn state(&self) -> &[f32] {
        &self.state
    }
}

impl PhysicalDevice for LoopbackDevice {
    fn dimension(&self) -> usize {
        self.state.len()
    }

    fn read_sensors(&mut self, output: &mut [f32]) -> Result<(), IoError> {
        if output.len() != self.state.len() {
            return Err(IoError::DimensionMismatch);
        }
        output.copy_from_slice(&self.state);
        Ok(())
    }

    fn write_actuators(&mut self, actions: &[f32]) -> Result<(), IoError> {
        if actions.len() != self.state.len() || actions.iter().any(|value| !value.is_finite()) {
            return Err(IoError::DimensionMismatch);
        }
        for (state, &action) in self.state.iter_mut().zip(actions) {
            *state = self.response.mul_add(action - *state, *state);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use argon_kernel::{PersistentGridConfig, PlasticityConfig};

    #[test]
    fn emulated_device_executes_hardware_loop() -> Result<(), Box<dyn std::error::Error>> {
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
        let mut coordinator = ArgonCoordinator::new(2, &grid, plasticity, 4)?;
        let device = LoopbackDevice::new(8, 0.2)?;
        let mut device_loop = DeviceLoop::new(device, &coordinator)?;
        for _ in 0..100 {
            let telemetry = device_loop.tick(&mut coordinator)?;
            assert!(telemetry.mean_goodness.is_finite());
        }
        assert!(device_loop
            .device()
            .state()
            .iter()
            .all(|value| value.is_finite()));
        Ok(())
    }
}
