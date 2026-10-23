//! Independent double-precision reference plant for numerical validation.

use crate::{MediumConfig, MediumError};

/// Double-precision periodic oscillator lattice.
#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceMedium {
    config: MediumConfig,
    positions: Vec<f64>,
    velocities: Vec<f64>,
    accelerations: Vec<f64>,
    next_accelerations: Vec<f64>,
}

impl ReferenceMedium {
    /// Constructs a zero-state reference lattice.
    ///
    /// # Errors
    /// Returns [`MediumError`] when the shared configuration is invalid.
    pub fn new(config: MediumConfig) -> Result<Self, MediumError> {
        // Construction through the production medium centralizes validation.
        let _validated = crate::CoupledOscillatorMedium::new(config)?;
        Ok(Self {
            config,
            positions: vec![0.0; config.dimension],
            velocities: vec![0.0; config.dimension],
            accelerations: vec![0.0; config.dimension],
            next_accelerations: vec![0.0; config.dimension],
        })
    }

    /// Replaces the state from matching finite slices.
    ///
    /// # Errors
    /// Returns [`MediumError`] on invalid dimensions or non-finite values.
    pub fn set_state(&mut self, positions: &[f64], velocities: &[f64]) -> Result<(), MediumError> {
        if positions.len() != self.config.dimension
            || velocities.len() != self.config.dimension
            || positions
                .iter()
                .chain(velocities)
                .any(|value| !value.is_finite())
        {
            return Err(MediumError::InvalidConfiguration(
                "reference state must be finite and dimensionally compatible",
            ));
        }
        self.positions.copy_from_slice(positions);
        self.velocities.copy_from_slice(velocities);
        Ok(())
    }

    /// Advances one split-damping Velocity Verlet step.
    ///
    /// # Errors
    /// Returns [`MediumError`] for invalid torque dimensions or values.
    pub fn advance(&mut self, torques: &[f64]) -> Result<(), MediumError> {
        if torques.len() != self.config.dimension || torques.iter().any(|value| !value.is_finite())
        {
            return Err(MediumError::InvalidConfiguration(
                "reference torques must be finite and dimensionally compatible",
            ));
        }
        let dt = f64::from(self.config.time_step);
        let damping = f64::from(self.config.damping);
        let damping_half = (-0.5 * damping * dt).exp();
        for velocity in &mut self.velocities {
            *velocity *= damping_half;
        }
        acceleration(
            &self.config,
            &self.positions,
            torques,
            &mut self.accelerations,
        );
        for ((position, velocity), &acceleration) in self
            .positions
            .iter_mut()
            .zip(&mut self.velocities)
            .zip(&self.accelerations)
        {
            *velocity = (0.5 * dt).mul_add(acceleration, *velocity);
            *position = dt.mul_add(*velocity, *position);
        }
        acceleration(
            &self.config,
            &self.positions,
            torques,
            &mut self.next_accelerations,
        );
        for (velocity, &acceleration) in self.velocities.iter_mut().zip(&self.next_accelerations) {
            *velocity = (0.5 * dt).mul_add(acceleration, *velocity) * damping_half;
        }
        std::mem::swap(&mut self.accelerations, &mut self.next_accelerations);
        Ok(())
    }

    /// Returns reference angular positions.
    pub fn positions(&self) -> &[f64] {
        &self.positions
    }

    /// Returns reference angular velocities.
    pub fn velocities(&self) -> &[f64] {
        &self.velocities
    }
}

fn acceleration(config: &MediumConfig, positions: &[f64], torques: &[f64], output: &mut [f64]) {
    let dimension = config.dimension;
    let coupling = f64::from(config.coupling);
    for index in 0..dimension {
        let left = positions[(index + dimension - 1) % dimension];
        let center = positions[index];
        let right = positions[(index + 1) % dimension];
        output[index] =
            coupling.mul_add((left - center) + (right - center), -center.sin()) + torques[index];
    }
}
