//! Heterogeneous graph-lattice plant and deterministic sensor/actuator effects.

use crate::MediumError;

/// Spring connectivity for a heterogeneous lattice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Topology {
    /// Periodic one-dimensional ring.
    Ring,
    /// Free or fixed-end one-dimensional chain.
    Chain {
        /// Connect both endpoints to zero-angle anchors.
        fixed_ends: bool,
    },
    /// Non-periodic row-major rectangular grid.
    Grid {
        /// Number of sites in one row.
        width: usize,
    },
    /// Caller-defined undirected graph edges.
    Arbitrary(Vec<(usize, usize)>),
}

/// Physical coefficients for one oscillator site.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SiteParameters {
    /// Point mass.
    pub mass: f32,
    /// Pendulum length.
    pub length: f32,
    /// Linear damping rate.
    pub damping: f32,
    /// Absolute torque limit.
    pub torque_limit: f32,
}

impl Default for SiteParameters {
    fn default() -> Self {
        Self {
            mass: 1.0,
            length: 1.0,
            damping: 0.15,
            torque_limit: 0.05,
        }
    }
}

/// Deterministic observation corruption configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SensorEffects {
    /// Uniform noise half-width; zero disables noise.
    pub noise_amplitude: f32,
    /// Quantization interval; zero disables quantization.
    pub quantization_step: f32,
    /// Every Nth observation is replaced by zero; zero disables dropout.
    pub dropout_period: u64,
    /// Whole-tick sensor latency.
    pub latency_ticks: usize,
    /// Whole-tick actuator latency.
    pub actuator_latency_ticks: usize,
}

impl Default for SensorEffects {
    fn default() -> Self {
        Self {
            noise_amplitude: 0.0,
            quantization_step: 0.0,
            dropout_period: 0,
            latency_ticks: 0,
            actuator_latency_ticks: 0,
        }
    }
}

/// Preallocated heterogeneous oscillator graph.
#[repr(align(64))]
#[derive(Debug)]
pub struct GraphOscillatorMedium {
    time_step: f32,
    coupling: f32,
    position_scale: f32,
    velocity_scale: f32,
    positions: Vec<f32>,
    velocities: Vec<f32>,
    parameters: Vec<SiteParameters>,
    edges: Vec<(usize, usize)>,
    fixed_ends: bool,
    accelerations: Vec<f32>,
    next_accelerations: Vec<f32>,
    limited_torques: Vec<f32>,
    delayed_torques: Vec<f32>,
    actuator_latency: usize,
    actuator_cursor: usize,
    raw_observation: Vec<f32>,
    delayed_observations: Vec<f32>,
    sensor_latency: usize,
    sensor_cursor: usize,
    sensor_effects: SensorEffects,
    random_state: u64,
    ticks: u64,
}

impl GraphOscillatorMedium {
    /// Creates a heterogeneous graph plant and preallocates latency storage.
    ///
    /// # Errors
    /// Returns [`MediumError`] for invalid dimensions, topology, or effects.
    pub fn new(
        dimension: usize,
        time_step: f32,
        coupling: f32,
        topology: Topology,
        sensor_effects: SensorEffects,
    ) -> Result<Self, MediumError> {
        if dimension < 2
            || !time_step.is_finite()
            || time_step <= 0.0
            || !coupling.is_finite()
            || coupling < 0.0
            || !sensor_effects.noise_amplitude.is_finite()
            || sensor_effects.noise_amplitude < 0.0
            || !sensor_effects.quantization_step.is_finite()
            || sensor_effects.quantization_step < 0.0
        {
            return Err(MediumError::InvalidConfiguration(
                "graph medium parameters are invalid",
            ));
        }
        let (edges, fixed_ends) = build_edges(dimension, topology)?;
        let channels = dimension
            .checked_mul(2)
            .ok_or(MediumError::InvalidConfiguration(
                "graph dimension is too large",
            ))?;
        let latency_slots = sensor_effects.latency_ticks.saturating_add(1);
        let latency_len =
            channels
                .checked_mul(latency_slots)
                .ok_or(MediumError::InvalidConfiguration(
                    "sensor latency storage is too large",
                ))?;
        let actuator_slots = sensor_effects.actuator_latency_ticks.saturating_add(1);
        let actuator_len =
            dimension
                .checked_mul(actuator_slots)
                .ok_or(MediumError::InvalidConfiguration(
                    "actuator latency storage is too large",
                ))?;
        Ok(Self {
            time_step,
            coupling,
            position_scale: std::f32::consts::PI,
            velocity_scale: 4.0,
            positions: vec![0.0; dimension],
            velocities: vec![0.0; dimension],
            parameters: vec![SiteParameters::default(); dimension],
            edges,
            fixed_ends,
            accelerations: vec![0.0; dimension],
            next_accelerations: vec![0.0; dimension],
            limited_torques: vec![0.0; dimension],
            delayed_torques: vec![0.0; actuator_len],
            actuator_latency: actuator_slots,
            actuator_cursor: 0,
            raw_observation: vec![0.0; channels],
            delayed_observations: vec![0.0; latency_len],
            sensor_latency: latency_slots,
            sensor_cursor: 0,
            sensor_effects,
            random_state: 0x4d59_5df4_d0f3_3173,
            ticks: 0,
        })
    }

    /// Replaces one site's physical parameters.
    ///
    /// # Errors
    /// Returns [`MediumError`] for an invalid index or coefficient.
    pub fn configure_site(
        &mut self,
        index: usize,
        parameters: SiteParameters,
    ) -> Result<(), MediumError> {
        if index >= self.parameters.len()
            || !parameters.mass.is_finite()
            || parameters.mass <= 0.0
            || !parameters.length.is_finite()
            || parameters.length <= 0.0
            || !parameters.damping.is_finite()
            || parameters.damping < 0.0
            || !parameters.torque_limit.is_finite()
            || parameters.torque_limit < 0.0
        {
            return Err(MediumError::InvalidConfiguration(
                "site parameters are invalid",
            ));
        }
        self.parameters[index] = parameters;
        Ok(())
    }

    /// Adds a finite shock to one site.
    ///
    /// # Errors
    /// Returns [`MediumError`] for an invalid index or value.
    pub fn perturb(
        &mut self,
        index: usize,
        position_delta: f32,
        velocity_delta: f32,
    ) -> Result<(), MediumError> {
        if index >= self.positions.len()
            || !position_delta.is_finite()
            || !velocity_delta.is_finite()
        {
            return Err(MediumError::InvalidConfiguration(
                "graph perturbation is invalid",
            ));
        }
        self.positions[index] += position_delta;
        self.velocities[index] += velocity_delta;
        Ok(())
    }

    /// Integrates one tick and writes a delayed, noisy, quantized observation.
    ///
    /// This method performs no heap allocation.
    ///
    /// # Errors
    /// Returns [`MediumError`] for invalid torque or observation storage.
    pub fn step(
        &mut self,
        requested_torques: &[f32],
        observation: &mut [f32],
    ) -> Result<(), MediumError> {
        if requested_torques.len() != self.positions.len()
            || observation.len() != self.raw_observation.len()
            || requested_torques.iter().any(|value| !value.is_finite())
        {
            return Err(MediumError::InvalidConfiguration(
                "graph step dimensions or torques are invalid",
            ));
        }
        let dimension = self.positions.len();
        let write_offset = self.actuator_cursor * dimension;
        for (index, (&requested, parameters)) in
            requested_torques.iter().zip(&self.parameters).enumerate()
        {
            self.delayed_torques[write_offset + index] =
                requested.clamp(-parameters.torque_limit, parameters.torque_limit);
        }
        self.actuator_cursor = (self.actuator_cursor + 1) % self.actuator_latency;
        let read_offset = self.actuator_cursor * dimension;
        self.limited_torques
            .copy_from_slice(&self.delayed_torques[read_offset..read_offset + dimension]);
        integrate(self);
        self.extract_observation(observation);
        self.ticks = self.ticks.wrapping_add(1);
        Ok(())
    }

    /// Returns physical positions.
    pub fn positions(&self) -> &[f32] {
        &self.positions
    }

    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    fn extract_observation(&mut self, output: &mut [f32]) {
        for index in 0..self.positions.len() {
            self.raw_observation[index * 2] =
                (self.positions[index] / self.position_scale).clamp(-1.0, 1.0);
            self.raw_observation[index * 2 + 1] =
                (self.velocities[index] / self.velocity_scale).clamp(-1.0, 1.0);
        }
        let dropout = self.sensor_effects.dropout_period != 0
            && self
                .ticks
                .is_multiple_of(self.sensor_effects.dropout_period);
        for index in 0..self.raw_observation.len() {
            let mut value = if dropout {
                0.0
            } else {
                self.raw_observation[index]
            };
            if self.sensor_effects.noise_amplitude > 0.0 {
                self.random_state ^= self.random_state << 13;
                self.random_state ^= self.random_state >> 7;
                self.random_state ^= self.random_state << 17;
                let unit = (self.random_state as u32) as f32 / u32::MAX as f32;
                value = self
                    .sensor_effects
                    .noise_amplitude
                    .mul_add(unit.mul_add(2.0, -1.0), value);
            }
            if self.sensor_effects.quantization_step > 0.0 {
                value = (value / self.sensor_effects.quantization_step).round()
                    * self.sensor_effects.quantization_step;
            }
            self.raw_observation[index] = value.clamp(-1.0, 1.0);
        }
        let channels = self.raw_observation.len();
        let write_offset = self.sensor_cursor * channels;
        self.delayed_observations[write_offset..write_offset + channels]
            .copy_from_slice(&self.raw_observation);
        self.sensor_cursor = (self.sensor_cursor + 1) % self.sensor_latency;
        let read_offset = self.sensor_cursor * channels;
        output.copy_from_slice(&self.delayed_observations[read_offset..read_offset + channels]);
    }
}

fn integrate(medium: &mut GraphOscillatorMedium) {
    for (velocity, parameters) in medium.velocities.iter_mut().zip(&medium.parameters) {
        *velocity *= (-0.5 * parameters.damping * medium.time_step).exp();
    }
    accelerations(medium, false);
    let half_dt = 0.5 * medium.time_step;
    for ((position, velocity), &acceleration) in medium
        .positions
        .iter_mut()
        .zip(&mut medium.velocities)
        .zip(&medium.accelerations)
    {
        *velocity = half_dt.mul_add(acceleration, *velocity);
        *position = medium.time_step.mul_add(*velocity, *position);
    }
    accelerations(medium, true);
    for ((velocity, &acceleration), parameters) in medium
        .velocities
        .iter_mut()
        .zip(&medium.next_accelerations)
        .zip(&medium.parameters)
    {
        *velocity = half_dt.mul_add(acceleration, *velocity);
        *velocity *= (-0.5 * parameters.damping * medium.time_step).exp();
    }
    std::mem::swap(&mut medium.accelerations, &mut medium.next_accelerations);
}

fn accelerations(medium: &mut GraphOscillatorMedium, next: bool) {
    let output = if next {
        &mut medium.next_accelerations
    } else {
        &mut medium.accelerations
    };
    for (index, output_slot) in output.iter_mut().enumerate() {
        let parameters = medium.parameters[index];
        let inverse_inertia = 1.0 / (parameters.mass * parameters.length * parameters.length);
        *output_slot = (-1.0 / parameters.length).mul_add(
            medium.positions[index].sin(),
            medium.limited_torques[index] * inverse_inertia,
        );
    }
    for &(left, right) in &medium.edges {
        let displacement = medium.positions[right] - medium.positions[left];
        let force = medium.coupling * displacement;
        let left_parameters = medium.parameters[left];
        let right_parameters = medium.parameters[right];
        let left_inverse =
            1.0 / (left_parameters.mass * left_parameters.length * left_parameters.length);
        let right_inverse =
            1.0 / (right_parameters.mass * right_parameters.length * right_parameters.length);
        output[left] = force.mul_add(left_inverse, output[left]);
        output[right] = (-force).mul_add(right_inverse, output[right]);
    }
    if medium.fixed_ends {
        let last = medium.positions.len() - 1;
        for index in [0, last] {
            let parameters = medium.parameters[index];
            let inverse = 1.0 / (parameters.mass * parameters.length * parameters.length);
            output[index] =
                (-medium.coupling * medium.positions[index]).mul_add(inverse, output[index]);
        }
    }
}

fn build_edges(
    dimension: usize,
    topology: Topology,
) -> Result<(Vec<(usize, usize)>, bool), MediumError> {
    let (edges, fixed_ends) = match topology {
        Topology::Ring => (
            (0..dimension)
                .map(|index| (index, (index + 1) % dimension))
                .collect(),
            false,
        ),
        Topology::Chain { fixed_ends } => (
            (0..dimension - 1).map(|index| (index, index + 1)).collect(),
            fixed_ends,
        ),
        Topology::Grid { width } => {
            if width == 0 || !dimension.is_multiple_of(width) {
                return Err(MediumError::InvalidConfiguration(
                    "grid width must divide dimension",
                ));
            }
            let mut edges = Vec::with_capacity(dimension.saturating_mul(2));
            for index in 0..dimension {
                if index % width + 1 < width {
                    edges.push((index, index + 1));
                }
                if index + width < dimension {
                    edges.push((index, index + width));
                }
            }
            (edges, false)
        }
        Topology::Arbitrary(edges) => (edges, false),
    };
    if edges
        .iter()
        .any(|&(left, right)| left >= dimension || right >= dimension || left == right)
    {
        return Err(MediumError::InvalidConfiguration(
            "graph edges must reference distinct valid sites",
        ));
    }
    Ok((edges, fixed_ends))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_effects_and_heterogeneous_sites_remain_finite() -> Result<(), MediumError> {
        let mut medium = GraphOscillatorMedium::new(
            6,
            0.005,
            0.2,
            Topology::Grid { width: 3 },
            SensorEffects {
                noise_amplitude: 0.01,
                quantization_step: 0.001,
                dropout_period: 11,
                latency_ticks: 2,
                actuator_latency_ticks: 3,
            },
        )?;
        medium.configure_site(
            0,
            SiteParameters {
                mass: 2.0,
                length: 0.8,
                damping: 0.2,
                torque_limit: 0.1,
            },
        )?;
        medium.perturb(0, 1.0, 0.5)?;
        let mut observation = [0.0; 12];
        for _ in 0..1_000 {
            medium.step(&[0.2; 6], &mut observation)?;
            assert!(observation.iter().all(|value| value.is_finite()));
        }
        assert!(medium.positions().iter().all(|value| value.is_finite()));
        Ok(())
    }
}
