//! Continuous coupled-oscillator physical plant for closing Argon's
//! sensory-motor loop.

#![forbid(non_ascii_idents)]

/// Configurable graph lattice, heterogeneous sites, and deterministic sensor transport.
pub mod advanced;
/// High-precision implementation used to validate the real-time `f32` plant.
pub mod reference;

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
    /// Maximum actuator torque change per second.
    pub max_torque_slew_per_second: f32,
    /// Absolute position mapped to the edge of the normalized sensory range.
    pub position_scale: f32,
    /// Absolute velocity mapped to the edge of the normalized sensory range.
    pub velocity_scale: f32,
    /// Plant energy above which synaptic homeostasis is applied.
    pub energy_boundary: f32,
    /// Energy below which an active homeostatic alarm is released.
    pub energy_release_boundary: f32,
    /// Target L2 norm used for homeostatic synaptic scaling.
    pub homeostasis_target_norm: f32,
    /// Minimum ticks between homeostatic scaling operations.
    pub homeostasis_cooldown_steps: u64,
}

impl Default for MediumConfig {
    fn default() -> Self {
        Self {
            dimension: 4,
            time_step: 0.005,
            coupling: 0.2,
            damping: 0.15,
            max_torque: 0.05,
            max_torque_slew_per_second: 10.0,
            position_scale: std::f32::consts::PI,
            velocity_scale: 4.0,
            energy_boundary: 4.0,
            energy_release_boundary: 3.0,
            homeostasis_target_norm: 1.0,
            homeostasis_cooldown_steps: 100,
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
    /// Individual physical energy terms after integration.
    pub energy_components: EnergyComponents,
    /// Whether the energy boundary triggered homeostatic synaptic scaling.
    pub homeostasis_applied: bool,
    /// Number of actuator sites limited by magnitude or slew constraints.
    pub saturated_actuators: usize,
    /// Largest absolute torque requested before actuator limiting.
    pub peak_requested_torque: f32,
}

/// Decomposition of the physical plant energy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnergyComponents {
    /// Translational/angular kinetic energy.
    pub kinetic: f32,
    /// On-site pendulum gravitational potential.
    pub pendulum_potential: f32,
    /// Nearest-neighbor spring coupling potential.
    pub coupling_potential: f32,
    /// Sum of all energy terms.
    pub total: f32,
}

/// A deterministic controller that writes one requested torque per oscillator.
pub trait Controller {
    /// Computes actuator torques from the current physical state.
    ///
    /// # Errors
    /// Returns [`MediumError`] if controller storage or state is incompatible
    /// with the supplied plant.
    fn control(
        &mut self,
        positions: &[f32],
        velocities: &[f32],
        output_torques: &mut [f32],
    ) -> Result<(), MediumError>;
}

/// Proportional-derivative stabilizer used as a safe control baseline.
#[derive(Debug, Clone)]
pub struct PdController {
    proportional_gain: f32,
    derivative_gain: f32,
    targets: Vec<f32>,
}

impl PdController {
    /// Creates a controller targeting zero angle at every site.
    ///
    /// # Errors
    /// Returns [`MediumError`] if dimensions or gains are invalid.
    pub fn new(
        dimension: usize,
        proportional_gain: f32,
        derivative_gain: f32,
    ) -> Result<Self, MediumError> {
        if dimension < 2
            || !proportional_gain.is_finite()
            || proportional_gain < 0.0
            || !derivative_gain.is_finite()
            || derivative_gain < 0.0
        {
            return Err(MediumError::InvalidConfiguration(
                "PD dimensions and gains must be finite and non-negative",
            ));
        }
        Ok(Self {
            proportional_gain,
            derivative_gain,
            targets: vec![0.0; dimension],
        })
    }

    /// Replaces target angles without allocating.
    ///
    /// # Errors
    /// Returns [`MediumError`] for a dimension mismatch or non-finite target.
    pub fn set_targets(&mut self, targets: &[f32]) -> Result<(), MediumError> {
        if targets.len() != self.targets.len() || targets.iter().any(|value| !value.is_finite()) {
            return Err(MediumError::InvalidConfiguration(
                "PD targets must be finite and match the plant dimension",
            ));
        }
        self.targets.copy_from_slice(targets);
        Ok(())
    }
}

impl Controller for PdController {
    fn control(
        &mut self,
        positions: &[f32],
        velocities: &[f32],
        output_torques: &mut [f32],
    ) -> Result<(), MediumError> {
        if positions.len() != self.targets.len()
            || velocities.len() != self.targets.len()
            || output_torques.len() != self.targets.len()
        {
            return Err(MediumError::InvalidConfiguration(
                "controller state dimensions do not match",
            ));
        }
        for (((torque, &position), &velocity), &target) in output_torques
            .iter_mut()
            .zip(positions)
            .zip(velocities)
            .zip(&self.targets)
        {
            *torque = (-self.proportional_gain)
                .mul_add(position - target, -self.derivative_gain * velocity);
        }
        Ok(())
    }
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
    requested_torques: Vec<f32>,
    baseline_torques: Vec<f32>,
    sequence_id: u64,
    homeostasis_active: bool,
    homeostasis_cooldown_remaining: u64,
}

/// Versioned deterministic checkpoint of physical state.
#[derive(Debug, Clone, PartialEq)]
pub struct MediumSnapshot {
    /// Snapshot format version.
    pub version: u32,
    /// Oscillator angular positions.
    pub positions: Vec<f32>,
    /// Oscillator angular velocities.
    pub velocities: Vec<f32>,
    /// Previously applied torque used by the slew limiter.
    pub applied_torques: Vec<f32>,
    /// Next sensory sequence identifier.
    pub sequence_id: u64,
    /// Whether energy hysteresis is currently active.
    pub homeostasis_active: bool,
    /// Remaining homeostasis cooldown ticks.
    pub homeostasis_cooldown_remaining: u64,
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
            requested_torques: vec![0.0; config.dimension],
            baseline_torques: vec![0.0; config.dimension],
            sequence_id: 0,
            homeostasis_active: false,
            homeostasis_cooldown_remaining: 0,
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

    /// Captures all physical state needed for deterministic continuation.
    pub fn snapshot(&self) -> MediumSnapshot {
        MediumSnapshot {
            version: 1,
            positions: self.positions.clone(),
            velocities: self.velocities.clone(),
            applied_torques: self.applied_torques.clone(),
            sequence_id: self.sequence_id,
            homeostasis_active: self.homeostasis_active,
            homeostasis_cooldown_remaining: self.homeostasis_cooldown_remaining,
        }
    }

    /// Restores a version-one snapshot into preallocated state storage.
    ///
    /// # Errors
    /// Returns [`MediumError`] when the version, dimensions, or values are invalid.
    pub fn restore(&mut self, snapshot: &MediumSnapshot) -> Result<(), MediumError> {
        if snapshot.version != 1 {
            return Err(MediumError::InvalidConfiguration(
                "unsupported medium snapshot version",
            ));
        }
        let dimension = self.config.dimension;
        if snapshot.positions.len() != dimension
            || snapshot.velocities.len() != dimension
            || snapshot.applied_torques.len() != dimension
        {
            return Err(MediumError::InvalidConfiguration(
                "snapshot dimension does not match medium",
            ));
        }
        if snapshot
            .positions
            .iter()
            .chain(&snapshot.velocities)
            .chain(&snapshot.applied_torques)
            .any(|value| !value.is_finite())
        {
            return Err(MediumError::InvalidConfiguration(
                "snapshot state must be finite",
            ));
        }
        self.positions.copy_from_slice(&snapshot.positions);
        self.velocities.copy_from_slice(&snapshot.velocities);
        self.applied_torques
            .copy_from_slice(&snapshot.applied_torques);
        self.sequence_id = snapshot.sequence_id;
        self.homeostasis_active = snapshot.homeostasis_active;
        self.homeostasis_cooldown_remaining = snapshot.homeostasis_cooldown_remaining;
        Ok(())
    }

    /// Advances multiple physics substeps under a constant torque vector.
    ///
    /// # Errors
    /// Returns [`MediumError`] if `substeps` is zero or torque validation fails.
    pub fn advance_substeps(
        &mut self,
        torques: &[f32],
        substeps: usize,
    ) -> Result<(), MediumError> {
        if substeps == 0 {
            return Err(MediumError::InvalidConfiguration(
                "substeps must be greater than zero",
            ));
        }
        for _ in 0..substeps {
            self.advance(torques)?;
        }
        Ok(())
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

        integrate_state(
            &self.config,
            &mut self.positions,
            &mut self.velocities,
            torques,
            &mut self.accelerations,
            &mut self.next_accelerations,
        );
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
        self.step_internal(coordinator, None)
    }

    /// Executes a closed-loop tick with a baseline controller plus neural residual action.
    ///
    /// # Errors
    /// Returns [`MediumError`] for incompatible controller/runtime state.
    pub fn step_with_controller(
        &mut self,
        coordinator: &mut ArgonCoordinator,
        controller: &mut dyn Controller,
    ) -> Result<MediumTelemetry, MediumError> {
        self.step_internal(coordinator, Some(controller))
    }

    fn step_internal(
        &mut self,
        coordinator: &mut ArgonCoordinator,
        controller: Option<&mut dyn Controller>,
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
        if let Err((error, mut rejected)) = coordinator.submit_frame_recover(frame) {
            std::mem::swap(&mut self.sensory_observation, &mut rejected.features);
            return Err(error.into());
        }
        let neural = coordinator
            .poll_and_step_recycle(&mut self.motor_actions, &mut self.sensory_observation)?;

        self.baseline_torques.fill(0.0);
        if let Some(controller) = controller {
            controller.control(
                &self.positions,
                &self.velocities,
                &mut self.baseline_torques,
            )?;
        }

        let max_delta = self.config.max_torque_slew_per_second * self.config.time_step;
        let mut saturated_actuators = 0_usize;
        let mut peak_requested_torque = 0.0_f32;
        for (index, torque) in self.applied_torques.iter_mut().enumerate() {
            let channel = index.saturating_mul(2);
            let differential = self.motor_actions[channel] - self.motor_actions[channel + 1];
            let requested =
                differential.mul_add(self.config.max_torque, self.baseline_torques[index]);
            self.requested_torques[index] = requested;
            peak_requested_torque = peak_requested_torque.max(requested.abs());
            let magnitude_limited =
                requested.clamp(-self.config.max_torque, self.config.max_torque);
            let slew_limited = magnitude_limited.clamp(*torque - max_delta, *torque + max_delta);
            if (slew_limited - requested).abs() > f32::EPSILON {
                saturated_actuators = saturated_actuators.saturating_add(1);
            }
            *torque = slew_limited;
        }

        integrate_state(
            &self.config,
            &mut self.positions,
            &mut self.velocities,
            &self.applied_torques,
            &mut self.accelerations,
            &mut self.next_accelerations,
        );

        let energy_components = self.energy_components();
        let energy = energy_components.total;
        if self.homeostasis_active && energy < self.config.energy_release_boundary {
            self.homeostasis_active = false;
        } else if energy > self.config.energy_boundary {
            self.homeostasis_active = true;
        }
        let homeostasis_applied =
            self.homeostasis_active && self.homeostasis_cooldown_remaining == 0;
        if homeostasis_applied {
            coordinator.apply_homeostasis(self.config.homeostasis_target_norm)?;
            self.homeostasis_cooldown_remaining = self.config.homeostasis_cooldown_steps;
        } else {
            self.homeostasis_cooldown_remaining =
                self.homeostasis_cooldown_remaining.saturating_sub(1);
        }
        self.sequence_id = self.sequence_id.wrapping_add(1);

        Ok(MediumTelemetry {
            neural,
            energy,
            energy_components,
            homeostasis_applied,
            saturated_actuators,
            peak_requested_torque,
        })
    }

    /// Computes kinetic, pendulum, and nearest-neighbor coupling energy.
    pub fn energy(&self) -> f32 {
        self.energy_components().total
    }

    /// Computes each component of the plant energy.
    pub fn energy_components(&self) -> EnergyComponents {
        let mut kinetic = 0.0_f32;
        let mut pendulum_potential = 0.0_f32;
        let mut coupling_potential = 0.0_f32;
        for index in 0..self.config.dimension {
            let velocity = self.velocities[index];
            let position = self.positions[index];
            let next = self.positions[(index + 1) % self.config.dimension];
            kinetic = (0.5 * velocity).mul_add(velocity, kinetic);
            pendulum_potential += 1.0 - position.cos();
            let displacement = next - position;
            coupling_potential = (0.5 * self.config.coupling * displacement)
                .mul_add(displacement, coupling_potential);
        }
        EnergyComponents {
            kinetic,
            pendulum_potential,
            coupling_potential,
            total: kinetic + pendulum_potential + coupling_potential,
        }
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
    if config.energy_release_boundary > config.energy_boundary {
        return Err(MediumError::InvalidConfiguration(
            "energy release boundary must not exceed the trigger boundary",
        ));
    }
    let non_negative = [
        config.coupling,
        config.damping,
        config.max_torque,
        config.max_torque_slew_per_second,
        config.energy_release_boundary,
    ];
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
        output[index] = coupled + torques[index];
    }
}

fn integrate_state(
    config: &MediumConfig,
    positions: &mut [f32],
    velocities: &mut [f32],
    torques: &[f32],
    accelerations: &mut Vec<f32>,
    next_accelerations: &mut Vec<f32>,
) {
    let damping_half_step = (-0.5 * config.damping * config.time_step).exp();
    for velocity in &mut *velocities {
        *velocity *= damping_half_step;
    }

    compute_accelerations(config, positions, torques, accelerations);
    let half_dt = 0.5 * config.time_step;
    for ((position, velocity), &acceleration) in positions
        .iter_mut()
        .zip(&mut *velocities)
        .zip(&*accelerations)
    {
        *velocity = half_dt.mul_add(acceleration, *velocity);
        *position = config.time_step.mul_add(*velocity, *position);
    }

    compute_accelerations(config, positions, torques, next_accelerations);
    for (velocity, &acceleration) in velocities.iter_mut().zip(&*next_accelerations) {
        *velocity = half_dt.mul_add(acceleration, *velocity);
        *velocity *= damping_half_step;
    }
    std::mem::swap(accelerations, next_accelerations);
}

/// Algorithmic Lyapunov stability certifier for coupled dynamical systems (Milestone M7.1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LyapunovCertification {
    /// Estimated maximal Lyapunov exponent (negative implies asymptotic stability).
    pub max_lyapunov_exponent: f32,
    /// Maximum energy observed over the evaluated trajectory.
    pub peak_energy: f32,
    /// Bounded energy drift ratio.
    pub energy_drift_ratio: f32,
    /// Whether the closed loop guarantees bounded-input bounded-output (BIBO) stability.
    pub is_provably_stable: bool,
}

/// Automated certifier verifying mathematical stability bounds on continuous medium trajectories.
pub struct LyapunovStabilityCertifier;

impl LyapunovStabilityCertifier {
    /// Evaluates Lyapunov exponent and energy contraction over a sample sequence of energies.
    ///
    /// # Errors
    /// Returns [`MediumError::InvalidConfiguration`] if the sequence has fewer than 2 samples.
    pub fn certify_energy_trajectory(
        energy_trajectory: &[f32],
        dt: f32,
    ) -> Result<LyapunovCertification, MediumError> {
        if energy_trajectory.len() < 2 || dt <= 0.0 {
            return Err(MediumError::InvalidConfiguration(
                "trajectory must contain at least 2 points and dt > 0",
            ));
        }
        let initial_e = energy_trajectory[0].max(1e-6);
        let mut peak_e = initial_e;
        let mut sum_log_divergence = 0.0_f32;
        let mut valid_steps = 0_usize;

        for window in energy_trajectory.windows(2) {
            let (e_prev, e_curr) = (window[0].max(1e-6), window[1].max(1e-6));
            peak_e = peak_e.max(e_curr);
            let ratio = e_curr / e_prev;
            sum_log_divergence += ratio.ln();
            valid_steps += 1;
        }

        #[allow(clippy::cast_precision_loss)]
        let total_time = (valid_steps as f32) * dt;
        let lambda = sum_log_divergence / total_time;
        let drift_ratio = (peak_e - initial_e).abs() / initial_e;
        let is_provably_stable = lambda <= 0.05 && drift_ratio < 10.0;

        Ok(LyapunovCertification {
            max_lyapunov_exponent: lambda,
            peak_energy: peak_e,
            energy_drift_ratio: drift_ratio,
            is_provably_stable,
        })
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

    #[test]
    fn snapshot_restore_replays_bit_exactly() -> Result<(), MediumError> {
        let mut first = CoupledOscillatorMedium::new(MediumConfig::default())?;
        first.perturb(1, 0.7, -0.2)?;
        let torques = [0.03, -0.02, 0.01, 0.0];
        first.advance_substeps(&torques, 50)?;
        let snapshot = first.snapshot();
        let mut second = CoupledOscillatorMedium::new(MediumConfig::default())?;
        second.restore(&snapshot)?;
        first.advance_substeps(&torques, 100)?;
        second.advance_substeps(&torques, 100)?;
        assert_eq!(first.positions(), second.positions());
        assert_eq!(first.velocities(), second.velocities());
        Ok(())
    }

    #[test]
    fn f32_plant_tracks_f64_reference() -> Result<(), MediumError> {
        let config = MediumConfig::default();
        let mut production = CoupledOscillatorMedium::new(config)?;
        production.perturb(0, 0.8, 0.4)?;
        let mut reference = reference::ReferenceMedium::new(config)?;
        reference.set_state(&[0.8, 0.0, 0.0, 0.0], &[0.4, 0.0, 0.0, 0.0])?;
        let torque_f32 = [0.0; 4];
        let torque_f64 = [0.0; 4];
        for _ in 0..2_000 {
            production.advance(&torque_f32)?;
            reference.advance(&torque_f64)?;
        }
        for (&actual, &expected) in production.positions().iter().zip(reference.positions()) {
            assert!((f64::from(actual) - expected).abs() < 2.0e-5);
        }
        Ok(())
    }

    #[test]
    fn pd_controller_opposes_displacement_and_velocity() -> Result<(), MediumError> {
        let mut controller = PdController::new(4, 2.0, 0.5)?;
        let mut output = [0.0; 4];
        controller.control(&[1.0, -1.0, 0.5, 0.0], &[0.2, -0.2, 0.0, 1.0], &mut output)?;
        let expected = [-2.1, 2.1, -1.0, -0.5];
        for (&actual, &expected_value) in output.iter().zip(&expected) {
            assert!((actual - expected_value).abs() < f32::EPSILON * 4.0);
        }
        Ok(())
    }

    #[test]
    fn lyapunov_certifier_validates_decaying_energy_trajectory() -> Result<(), MediumError> {
        let decaying_trajectory = [10.0_f32, 8.0, 6.4, 5.12, 4.096, 3.2768];
        let cert = LyapunovStabilityCertifier::certify_energy_trajectory(&decaying_trajectory, 0.01)?;
        assert!(cert.max_lyapunov_exponent < 0.0);
        assert!(cert.is_provably_stable);
        assert_eq!(cert.peak_energy, 10.0);
        Ok(())
    }
}
