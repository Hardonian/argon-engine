//! Deterministic stability sweep across integration and lattice coefficients.

use argon_medium::{CoupledOscillatorMedium, MediumConfig, MediumError};

#[test]
fn parameter_matrix_remains_finite_and_damped() -> Result<(), MediumError> {
    for time_step in [0.001, 0.005, 0.01] {
        for coupling in [0.0, 0.2, 0.8] {
            for damping in [0.05, 0.15, 0.5] {
                let mut medium = CoupledOscillatorMedium::new(MediumConfig {
                    time_step,
                    coupling,
                    damping,
                    energy_boundary: 1_000.0,
                    energy_release_boundary: 900.0,
                    ..MediumConfig::default()
                })?;
                medium.perturb(0, 1.5, 2.0)?;
                medium.perturb(2, -0.8, -1.0)?;
                let initial = medium.energy();
                for _ in 0..5_000 {
                    medium.advance(&[0.0; 4])?;
                    assert!(medium.energy().is_finite());
                }
                assert!(medium.energy() < initial);
            }
        }
    }
    Ok(())
}
