//! Explicit backend selection and capability reporting.

#![forbid(non_ascii_idents)]

use argon_runtime::{ArgonCoordinator, RuntimeError, TickTelemetry};
use std::process::Command;

/// Execution engine contract shared by synchronous backends.
pub trait ExecutionBackend {
    /// Returns the backend's required feature dimension.
    fn dimension(&self) -> usize;

    /// Executes one borrowed-buffer tick.
    ///
    /// # Errors
    /// Returns [`RuntimeError`] for invalid inputs or backend execution errors.
    fn step(
        &mut self,
        sequence_id: u64,
        polarity: f32,
        features: &[f32],
        output: &mut [f32],
    ) -> Result<TickTelemetry, RuntimeError>;
}

/// Normative scalar/portable-SIMD CPU backend.
pub struct CpuBackend {
    coordinator: ArgonCoordinator,
}

impl CpuBackend {
    /// Wraps an initialized coordinator.
    pub const fn new(coordinator: ArgonCoordinator) -> Self {
        Self { coordinator }
    }

    /// Returns the wrapped coordinator for checkpointing and telemetry.
    pub const fn coordinator(&self) -> &ArgonCoordinator {
        &self.coordinator
    }
}

impl ExecutionBackend for CpuBackend {
    fn dimension(&self) -> usize {
        self.coordinator.tile_dim()
    }

    fn step(
        &mut self,
        sequence_id: u64,
        polarity: f32,
        features: &[f32],
        output: &mut [f32],
    ) -> Result<TickTelemetry, RuntimeError> {
        self.coordinator
            .step_borrowed(sequence_id, polarity, features, output)
    }
}

/// CUDA driver and device information discovered without selecting the backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaCapabilities {
    /// NVIDIA driver version reported by `nvidia-smi`.
    pub driver_version: String,
    /// Names of visible CUDA devices.
    pub devices: Vec<String>,
}

impl CudaCapabilities {
    /// Probes the local NVIDIA management interface.
    ///
    /// Absence is represented by `None` and never silently falls back while
    /// claiming CUDA execution.
    pub fn probe() -> Option<Self> {
        let output = Command::new("nvidia-smi")
            .args([
                "--query-gpu=driver_version,name",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let stdout = String::from_utf8(output.stdout).ok()?;
        let mut driver_version = None;
        let mut devices = Vec::new();
        for line in stdout.lines().filter(|line| !line.trim().is_empty()) {
            let (driver, name) = line.split_once(',')?;
            driver_version.get_or_insert_with(|| driver.trim().to_owned());
            devices.push(name.trim().to_owned());
        }
        Some(Self {
            driver_version: driver_version?,
            devices,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use argon_kernel::{PersistentGridConfig, PlasticityConfig};

    #[test]
    fn cpu_backend_executes_borrowed_tick() -> Result<(), RuntimeError> {
        let grid = PersistentGridConfig {
            grid_dim_x: 1,
            block_dim_x: 32,
            shared_mem_bytes: 4_096,
            tile_dim: 4,
        };
        let plasticity = PlasticityConfig {
            learning_rate: 0.01,
            goodness_threshold: 1.0,
            weight_decay: 0.001,
            phase_polarity: 1.0,
        };
        let coordinator = ArgonCoordinator::new(1, &grid, plasticity, 4)?;
        let mut backend = CpuBackend::new(coordinator);
        let mut output = [0.0; 4];
        let telemetry = backend.step(9, 1.0, &[0.1, 0.2, 0.3, 0.4], &mut output)?;
        assert_eq!(telemetry.sequence_id, 9);
        assert!(output.iter().all(|value| value.is_finite()));
        Ok(())
    }
}
