//! Continuous coupled-oscillator physical plant for closing Argon's
//! sensory-motor loop.

#![forbid(non_ascii_idents)]

use argon_runtime::{ArgonCoordinator, ControlFrame, RuntimeError, TickTelemetry};
use std::fmt;

/// Configuration for a ring of non-linear coupled oscillators.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MediumConfig {
    /// Number of oscillator sites. The coordinator dimension must be twice this value.
    pub dimension: usize,
    /// Integration time step in seconds.
    pub time_step: f32,
    /// Nearest-neighbor spring coupling coefficient.
    pub coupling: f32,
    /// Linear velocity damping coefficient.
    pub damping: f32,
    /// Maximum absolute actuator torque per site.
    pub max_torque: f32,
    /// Absolute position mapped to the edge of the normalized sensory range.
    pub position_scale: f32,
    /// Absolute velocity mapped to the edge of the normalized sensory range.
    pub velocity_scale: f32,
    /// Plant energy above which synaptic homeostasis is applied.
    pub energy_boundary: f32,
    /// Target L2 norm used for homeostatic synaptic scaling.
    pub homeostasis_target_norm: f32,
}

impl Default for MediumConfig {
    fn default() -> Self {
        Self {
            dimension: 4,
            time_step: 0.005,
            coupling: 0.2,
            damping: 0.15,
            max_torque: 0.05,
            position_scale: std::f32::consts::PI,
            velocity_scale: 4.0,
            energy_boundary: 4.0,
            homeostasis_target_norm: 1.0,
        }
    }
}

/// Errors emitted by the physical medium or its attached coordinator.
#[derive(Debug)]
pub enum MediumError {
    /// A physical or dimensional configuration is invalid.
    InvalidConfiguration(&'static str),
    /// The coordinator rejected a frame or neural operation.
    Runtime(RuntimeError),
}

impl fmt::Display for MediumError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration(message) => {
                write!(f, "Invalid physical medium configuration: {message}")
            }
            Self::Runtime(error) => write!(f, "Physical medium runtime error: {error}"),
        }
    }
}

impl std::error::Error for MediumError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidConfiguration(_) => None,
            Self::Runtime(error) => Some(error),
        }
    }
}

impl From<RuntimeError> for MediumError {
    fn from(error: RuntimeError) -> Self {
        Self::Runtime(error)
    }
}

/// Telemetry emitted by a complete sensory, neural, actuator, and physics tick.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MediumTelemetry {
    /// Neural execution telemetry for this tick.
    pub neural: TickTelemetry,
    /// Hamiltonian-like kinetic, pendulum, and coupling energy after integration.
    pub energy: f32,
    /// Whether the energy boundary triggered homeostatic synaptic scaling.
    pub homeostasis_applied: bool,
}

/// Cache-line-aligned ring of pendula coupled by nearest-neighbor springs.
///
/// Each oscillator contributes two interleaved neural channels. During sensing
/// they contain normalized position and velocity. During actuation their
/// difference forms one signed motor command.
#[repr(align(64))]
#[derive(Debug)]
pub struct CoupledOscillatorMedium {
    config: MediumConfig,
    positions: Vec<f32>,
    velocities: Vec<f32>,
    accelerations: Vec<f32>,
    next_accelerations: Vec<f32>,
    sensory_observation: Vec<f32>,
    motor_actions: Vec<f32>,
    applied_torques: Vec<f32>,
    sequence_id: u64,
}

impl CoupledOscillatorMedium {
    /// Allocates all state and scratch storage required by the real-time loop.
    ///
    /// # Errors
    /// Returns [`MediumError::InvalidConfiguration`] for non-finite, negative,
    /// zero, or overflowing parameters that cannot define a stable plant.
    pub fn new(config: MediumConfig) -> Result<Self, MediumError> {
        validate_config(config)?;
        let feature_dim = config
            .dimension
            .checked_mul(2)
            .ok_or(MediumError::InvalidConfiguration("dimension is too large"))?;
        Ok(Self {
            config,
            positions: vec![0.0; config.dimension],
            velocities: vec![0.0; config.dimension],
            accelerations: vec![0.0; config.dimension],
            next_accelerations: vec![0.0; config.dimension],
            sensory_observation: vec![0.0; feature_dim],
            motor_actions: vec![0.0; feature_dim],
            applied_torques: vec![0.0; config.dimension],
            sequence_id: 0,
        })
    }

    /// Returns the immutable physical configuration.
    pub const fn config(&self) -> &MediumConfig {
        &self.config
    }

    /// Returns all angular positions in radians.
    pub fn positions(&self) -> &[f32] {
        &self.positions
    }

    /// Returns all angular velocities in radians per second.
    pub fn velocities(&self) -> &[f32] {
        &self.velocities
    }

    /// Adds a finite displacement and velocity impulse to one oscillator.
    ///
    /// # Errors
    /// Returns [`MediumError::InvalidConfiguration`] if the index is out of
    /// range or either impulse component is non-finite.
    pub fn perturb(
        &mut self,
        index: usize,
        position_delta: f32,
        velocity_delta: f32,
    ) -> Result<(), MediumError> {
        if !position_delta.is_finite() || !velocity_delta.is_finite() {
            return Err(MediumError::InvalidConfiguration(
                "perturbation components must be finite",
            ));
        }
        let position = self
            .positions
            .get_mut(index)
            .ok_or(MediumError::InvalidConfiguration(
                "perturbation index is out of bounds",
            ))?;
        let velocity = self
            .velocities
            .get_mut(index)
            .ok_or(MediumError::InvalidConfiguration(
                "perturbation index is out of bounds",
            ))?;
        *position += position_delta;
        *velocity += velocity_delta;
        Ok(())
    }

    /// Advances one kick-drift-kick Velocity Verlet step with supplied torques.
    ///
    /// No allocation occurs in this method.
    ///
    /// # Errors
    /// Returns [`MediumError::InvalidConfiguration`] when the torque slice has
    /// the wrong length or contains a non-finite value.
    pub fn advance(&mut self, torques: &[f32]) -> Result<(), MediumError> {
        if torques.len() != self.config.dimension {
            return Err(MediumError::InvalidConfiguration(
                "torque dimension must match the oscillator dimension",
            ));
        }
        if torques.iter().any(|value| !value.is_finite()) {
            return Err(MediumError::InvalidConfiguration(
                "all actuator torques must be finite",
            ));
        }

        compute_accelerations(
            &self.config,
            &self.positions,
            &self.velocities,
            torques,
            &mut self.accelerations,
        );
        let half_dt = 0.5 * self.config.time_step;
        for ((position, velocity), &acceleration) in self
            .positions
            .iter_mut()
            .zip(&mut self.velocities)
            .zip(&self.accelerations)
        {
            *velocity = half_dt.mul_add(acceleration, *velocity);
            *position = self.config.time_step.mul_add(*velocity, *position);
        }

        compute_accelerations(
            &self.config,
            &self.positions,
            &self.velocities,
            torques,
            &mut self.next_accelerations,
        );
        for (velocity, &acceleration) in self.velocities.iter_mut().zip(&self.next_accelerations) {
            *velocity = half_dt.mul_add(acceleration, *velocity);
        }
        std::mem::swap(&mut self.accelerations, &mut self.next_accelerations);
        Ok(())
    }

    /// Executes one allocation-free closed-loop sensory-to-actuator tick.
    ///
    /// # Errors
    /// Returns [`MediumError`] when the coordinator dimension does not match
    /// the medium or any runtime operation fails.
    pub fn step(
        &mut self,
        coordinator: &mut ArgonCoordinator,
    ) -> Result<MediumTelemetry, MediumError> {
        if coordinator.tile_dim() != self.sensory_observation.len() {
            return Err(MediumError::InvalidConfiguration(
                "coordinator tile dimension must equal twice the medium dimension",
            ));
        }

        for (index, (&position, &velocity)) in
            self.positions.iter().zip(&self.velocities).enumerate()
        {
            let channel = index.saturating_mul(2);
            self.sensory_observation[channel] =
                (position / self.config.position_scale).clamp(-1.0, 1.0);
            self.sensory_observation[channel + 1] =
                (velocity / self.config.velocity_scale).clamp(-1.0, 1.0);
        }

        let frame = ControlFrame {
            sequence_id: self.sequence_id,
            phase_polarity: 1.0,
            features: std::mem::take(&mut self.sensory_observation),
        };
        coordinator.submit_frame(frame)?;
        let neural = coordinator
            .poll_and_step_recycle(&mut self.motor_actions, &mut self.sensory_observation)?;

        for (index, torque) in self.applied_torques.iter_mut().enumerate() {
            let channel = index.saturating_mul(2);
            let differential = self.motor_actions[channel] - self.motor_actions[channel + 1];
            *torque = differential.clamp(-1.0, 1.0) * self.config.max_torque;
        }

        let config = self.config;
        compute_accelerations(
            &config,
            &self.positions,
            &self.velocities,
            &self.applied_torques,
            &mut self.accelerations,
        );
        let half_dt = 0.5 * config.time_step;
        for ((position, velocity), &acceleration) in self
            .positions
            .iter_mut()
            .zip(&mut self.velocities)
            .zip(&self.accelerations)
        {
            *velocity = half_dt.mul_add(acceleration, *velocity);
            *position = config.time_step.mul_add(*velocity, *position);
        }
        compute_accelerations(
            &config,
            &self.positions,
            &self.velocities,
            &self.applied_torques,
            &mut self.next_accelerations,
        );
        for (velocity, &acceleration) in self.velocities.iter_mut().zip(&self.next_accelerations) {
            *velocity = half_dt.mul_add(acceleration, *velocity);
        }
        std::mem::swap(&mut self.accelerations, &mut self.next_accelerations);

        let energy = self.energy();
        let homeostasis_applied = energy > self.config.energy_boundary;
        if homeostasis_applied {
            coordinator.apply_homeostasis(self.config.homeostasis_target_norm)?;
        }
        self.sequence_id = self.sequence_id.wrapping_add(1);

        Ok(MediumTelemetry {
            neural,
            energy,
            homeostasis_applied,
        })
    }

    /// Computes kinetic, pendulum, and nearest-neighbor coupling energy.
    pub fn energy(&self) -> f32 {
        let mut total = 0.0_f32;
        for index in 0..self.config.dimension {
            let velocity = self.velocities[index];
            let position = self.positions[index];
            let next = self.positions[(index + 1) % self.config.dimension];
            total = (0.5 * velocity).mul_add(velocity, total);
            total += 1.0 - position.cos();
            let displacement = next - position;
            total = (0.5 * self.config.coupling * displacement).mul_add(displacement, total);
        }
        total
    }
}

fn validate_config(config: MediumConfig) -> Result<(), MediumError> {
    if config.dimension < 2 {
        return Err(MediumError::InvalidConfiguration(
            "dimension must be at least two",
        ));
    }
    let positive = [
        config.time_step,
        config.position_scale,
        config.velocity_scale,
        config.energy_boundary,
        config.homeostasis_target_norm,
    ];
    if positive
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(MediumError::InvalidConfiguration(
            "scales, boundaries, target norm, and time step must be finite and positive",
        ));
    }
    let non_negative = [config.coupling, config.damping, config.max_torque];
    if non_negative
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(MediumError::InvalidConfiguration(
            "coupling, damping, and maximum torque must be finite and non-negative",
        ));
    }
    Ok(())
}

fn compute_accelerations(
    config: &MediumConfig,
    positions: &[f32],
    velocities: &[f32],
    torques: &[f32],
    output: &mut [f32],
) {
    let dimension = config.dimension;
    for index in 0..dimension {
        let left = positions[(index + dimension - 1) % dimension];
        let center = positions[index];
        let right = positions[(index + 1) % dimension];
        let neighbor_delta = (left - center) + (right - center);
        let coupled = config.coupling.mul_add(neighbor_delta, -center.sin());
        output[index] = (-config.damping).mul_add(velocities[index], coupled + torques[index]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conservative_verlet_has_bounded_energy_error() -> Result<(), MediumError> {
        let config = MediumConfig {
            damping: 0.0,
            max_torque: 0.0,
            energy_boundary: 100.0,
            ..MediumConfig::default()
        };
        let mut medium = CoupledOscillatorMedium::new(config)?;
        medium.perturb(0, 0.8, 0.4)?;
        let initial = medium.energy();
        let torques = [0.0; 4];
        let mut max_error = 0.0_f32;
        for _ in 0..20_000 {
            medium.advance(&torques)?;
            max_error = max_error.max((medium.energy() - initial).abs());
        }
        assert!(
            max_error / initial < 0.000_1,
            "relative error: {}",
            max_error / initial
        );
        Ok(())
    }
}
